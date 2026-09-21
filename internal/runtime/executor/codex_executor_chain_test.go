package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

// codexChainTurn1Completed finishes a turn whose only output is a function call,
// the exact shape that triggers "No tool call found for function call output"
// when the next incremental request is forwarded statelessly.
const codexChainTurn1Completed = `data: {"type":"response.completed","response":{"id":"resp_1","object":"response","created_at":0,"status":"completed","background":false,"error":null,"output":[{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"get_weather","arguments":"{}"}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`

const codexChainTurn2Completed = `data: {"type":"response.completed","response":{"id":"resp_2","object":"response","created_at":0,"status":"completed","background":false,"error":null,"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"It is sunny in SF."}]}],"usage":{"input_tokens":20,"output_tokens":8,"total_tokens":28}}}`

const codexChainTurn2ToolCallCompleted = `data: {"type":"response.completed","response":{"id":"resp_2","object":"response","created_at":0,"status":"completed","background":false,"error":null,"output":[{"type":"function_call","id":"fc_2","status":"completed","call_id":"call_2","name":"get_time","arguments":"{}"}],"usage":{"input_tokens":20,"output_tokens":5,"total_tokens":25}}}`

const codexChainTurn3Completed = `data: {"type":"response.completed","response":{"id":"resp_3","object":"response","created_at":0,"status":"completed","background":false,"error":null,"output":[{"type":"message","id":"msg_2","status":"completed","role":"assistant","content":[{"type":"output_text","text":"It is noon."}]}],"usage":{"input_tokens":30,"output_tokens":8,"total_tokens":38}}}`

const codexChainTurn1Request = `{"model":"gpt-5.4","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in SF?"}]}]}`

const codexChainTurn2Request = `{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`

func newCodexChainFixture(t *testing.T) (*CodexExecutor, *cliproxyauth.Auth, *[][]byte) {
	return newCodexChainScriptedFixture(t, nil)
}

// newCodexChainScriptedFixture serves the given SSE payloads per upstream
// request (1-indexed by turn; the last payload repeats for further turns) and
// captures every upstream request body.
func newCodexChainScriptedFixture(t *testing.T, scripted []string, cfg ...*config.Config) (*CodexExecutor, *cliproxyauth.Auth, *[][]byte) {
	t.Helper()
	helps.ClearCodexResponsesChainCache()
	t.Cleanup(helps.ClearCodexResponsesChainCache)

	var mu sync.Mutex
	var requests [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, errRead := io.ReadAll(r.Body)
		if errRead != nil {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		mu.Lock()
		requests = append(requests, body)
		turn := len(requests)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		payload := codexChainTurn2Completed
		if len(scripted) > 0 {
			payload = scripted[len(scripted)-1]
			if turn <= len(scripted) {
				payload = scripted[turn-1]
			}
		} else if turn == 1 {
			payload = codexChainTurn1Completed
		}
		_, _ = w.Write([]byte(payload + "\n\n"))
	}))
	t.Cleanup(server.Close)

	executorConfig := &config.Config{}
	if len(cfg) > 0 && cfg[0] != nil {
		executorConfig = cfg[0]
	}
	executor := NewCodexExecutor(executorConfig)
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL,
		"api_key":  "test",
	}}
	return executor, auth, &requests
}

func assertCodexChainRebuiltSecondRequest(t *testing.T, requests *[][]byte) {
	t.Helper()
	if requests == nil || len(*requests) != 2 {
		count := 0
		if requests != nil {
			count = len(*requests)
		}
		t.Fatalf("upstream requests = %d, want 2", count)
	}
	second := (*requests)[1]
	if gjson.GetBytes(second, "previous_response_id").Exists() {
		t.Fatalf("previous_response_id forwarded upstream: %s", second)
	}
	if got := gjson.GetBytes(second, "store"); got.Type != gjson.False {
		t.Fatalf("store = %v, want false", got.Raw)
	}
	items := gjson.GetBytes(second, "input").Array()
	if len(items) != 3 {
		t.Fatalf("input items = %d, want 3 (user, function_call, function_call_output): %s", len(items), second)
	}
	if got := items[0].Get("type").String(); got != "message" {
		t.Fatalf("items[0].type = %q, want message", got)
	}
	if got := items[0].Get("content.0.text").String(); got != "weather in SF?" {
		t.Fatalf("items[0].text = %q, want original user text", got)
	}
	if got := items[1].Get("type").String(); got != "function_call" {
		t.Fatalf("items[1].type = %q, want function_call", got)
	}
	if got := items[1].Get("call_id").String(); got != "call_1" {
		t.Fatalf("items[1].call_id = %q, want call_1", got)
	}
	if got := items[2].Get("type").String(); got != "function_call_output" {
		t.Fatalf("items[2].type = %q, want function_call_output", got)
	}
}

func TestCodexExecutorExecuteRebuildsPreviousResponseChain(t *testing.T) {
	executor, auth, requests := newCodexChainFixture(t)

	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(codexChainTurn1Request),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false})
	if err != nil {
		t.Fatalf("turn 1 Execute error: %v", err)
	}

	_, err = executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(codexChainTurn2Request),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false})
	if err != nil {
		t.Fatalf("turn 2 Execute error: %v", err)
	}

	assertCodexChainRebuiltSecondRequest(t, requests)
}

func TestCodexExecutorExecuteStreamRebuildsPreviousResponseChain(t *testing.T) {
	executor, auth, requests := newCodexChainFixture(t)

	drain := func(payload string) {
		t.Helper()
		result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
			Model:   "gpt-5.4",
			Payload: []byte(payload),
		}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: true})
		if err != nil {
			t.Fatalf("ExecuteStream error: %v", err)
		}
		for range result.Chunks {
		}
	}

	drain(codexChainTurn1Request)
	drain(codexChainTurn2Request)

	assertCodexChainRebuiltSecondRequest(t, requests)
}

func TestCodexExecutorExecuteChainCacheMissReturnsPreviousResponseNotFound(t *testing.T) {
	executor, auth, requests := newCodexChainFixture(t)

	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(`{"model":"gpt-5.4","previous_response_id":"resp_missing","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false})
	if err == nil {
		t.Fatalf("expected error for missing snapshot")
	}
	statusErr, ok := err.(interface{ StatusCode() int })
	if !ok {
		t.Fatalf("error %T lacks StatusCode", err)
	}
	if code := statusErr.StatusCode(); code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", code)
	}
	body := err.Error()
	if got := gjson.Get(body, "error.code").String(); got != "previous_response_not_found" {
		t.Fatalf("error.code = %q, want previous_response_not_found (body: %s)", got, body)
	}
	if got := gjson.Get(body, "error.param").String(); got != "previous_response_id" {
		t.Fatalf("error.param = %q, want previous_response_id", got)
	}
	// The broken request must not be forwarded upstream.
	if requests != nil && len(*requests) != 0 {
		t.Fatalf("upstream requests = %d, want 0", len(*requests))
	}
}

func TestCodexExecutorExecuteStreamChainCacheMissReturnsPreviousResponseNotFound(t *testing.T) {
	executor, auth, requests := newCodexChainFixture(t)

	_, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(`{"model":"gpt-5.4","previous_response_id":"resp_missing","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: true})
	if err == nil {
		t.Fatalf("expected error for missing snapshot")
	}
	statusErr, ok := err.(interface{ StatusCode() int })
	if !ok {
		t.Fatalf("error %T lacks StatusCode", err)
	}
	if code := statusErr.StatusCode(); code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", code)
	}
	if got := gjson.Get(err.Error(), "error.code").String(); got != "previous_response_not_found" {
		t.Fatalf("error.code = %q, want previous_response_not_found", got)
	}
	if requests != nil && len(*requests) != 0 {
		t.Fatalf("upstream requests = %d, want 0", len(*requests))
	}
}

func TestCodexExecutorExecuteChainAccumulatesAcrossMultipleHops(t *testing.T) {
	executor, auth, requests := newCodexChainScriptedFixture(t, []string{
		codexChainTurn1Completed,
		codexChainTurn2ToolCallCompleted,
		codexChainTurn3Completed,
	})

	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(codexChainTurn1Request),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false}); err != nil {
		t.Fatalf("turn 1 Execute error: %v", err)
	}
	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(codexChainTurn2Request),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false}); err != nil {
		t.Fatalf("turn 2 Execute error: %v", err)
	}
	turn3 := `{"model":"gpt-5.4","previous_response_id":"resp_2","input":[{"type":"function_call_output","call_id":"call_2","output":"noon"}]}`
	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(turn3),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false}); err != nil {
		t.Fatalf("turn 3 Execute error: %v", err)
	}

	if len(*requests) != 3 {
		t.Fatalf("upstream requests = %d, want 3", len(*requests))
	}
	third := (*requests)[2]
	if gjson.GetBytes(third, "previous_response_id").Exists() {
		t.Fatalf("previous_response_id forwarded upstream: %s", third)
	}
	items := gjson.GetBytes(third, "input").Array()
	if len(items) != 5 {
		t.Fatalf("input items = %d, want 5 accumulated items: %s", len(items), third)
	}
	wantTypes := []string{"message", "function_call", "function_call_output", "function_call", "function_call_output"}
	wantCallIDs := []string{"", "call_1", "call_1", "call_2", "call_2"}
	for i, want := range wantTypes {
		if got := items[i].Get("type").String(); got != want {
			t.Fatalf("items[%d].type = %q, want %q (%s)", i, got, want, third)
		}
	}
	for i, want := range wantCallIDs {
		if got := items[i].Get("call_id").String(); got != want {
			t.Fatalf("items[%d].call_id = %q, want %q (%s)", i, got, want, third)
		}
	}
}

func TestCodexExecutorExecuteSkipsRebuildForFullReplayClient(t *testing.T) {
	executor, auth, requests := newCodexChainFixture(t)

	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(codexChainTurn1Request),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false}); err != nil {
		t.Fatalf("turn 1 Execute error: %v", err)
	}

	fullReplay := `{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in SF?"}]},{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{}"},{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`
	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(fullReplay),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false}); err != nil {
		t.Fatalf("full replay Execute error: %v", err)
	}

	assertCodexChainRebuiltSecondRequest(t, requests)
}

func TestCodexExecutorExecuteStreamRecordsResponseDoneEvent(t *testing.T) {
	doneEvent := `data: {"type":"response.done","response":{"id":"resp_done","object":"response","created_at":0,"status":"completed","background":false,"error":null,"output":[{"type":"function_call","id":"fc_d","status":"completed","call_id":"call_d","name":"get_weather","arguments":"{}"}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`
	executor, auth, requests := newCodexChainScriptedFixture(t, []string{doneEvent, codexChainTurn2Completed})

	for _, payload := range []string{
		codexChainTurn1Request,
		`{"model":"gpt-5.4","previous_response_id":"resp_done","input":[{"type":"function_call_output","call_id":"call_d","output":"sunny"}]}`,
	} {
		result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
			Model:   "gpt-5.4",
			Payload: []byte(payload),
		}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: true})
		if err != nil {
			t.Fatalf("ExecuteStream error: %v", err)
		}
		for range result.Chunks {
		}
	}

	if len(*requests) != 2 {
		t.Fatalf("upstream requests = %d, want 2", len(*requests))
	}
	second := (*requests)[1]
	if gjson.GetBytes(second, "previous_response_id").Exists() {
		t.Fatalf("previous_response_id forwarded upstream: %s", second)
	}
	items := gjson.GetBytes(second, "input").Array()
	if len(items) != 3 {
		t.Fatalf("input items = %d, want 3 after recording response.done: %s", len(items), second)
	}
	if got := items[1].Get("call_id").String(); got != "call_d" {
		t.Fatalf("items[1].call_id = %q, want call_d", got)
	}
}

func TestCodexExecutorExecuteDoesNotRecordIncompleteResponse(t *testing.T) {
	incompleteEvent := `data: {"type":"response.incomplete","response":{"id":"resp_inc","object":"response","created_at":0,"status":"incomplete","background":false,"error":null,"output":[{"type":"message","id":"msg_i","status":"completed","role":"assistant","content":[{"type":"output_text","text":"partial"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`
	executor, auth, requests := newCodexChainScriptedFixture(t, []string{incompleteEvent, codexChainTurn2Completed})

	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(codexChainTurn1Request),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false}); err != nil {
		t.Fatalf("turn 1 Execute error: %v", err)
	}
	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(`{"model":"gpt-5.4","previous_response_id":"resp_inc","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false})
	if err == nil {
		t.Fatalf("expected previous_response_not_found for unanchored chain")
	}
	statusErr, ok := err.(interface{ StatusCode() int })
	if !ok {
		t.Fatalf("turn 2 error %T lacks StatusCode: %v", err, err)
	}
	if code := statusErr.StatusCode(); code != http.StatusBadRequest {
		t.Fatalf("turn 2 status = %d, want 400", code)
	}
	if got := gjson.Get(err.Error(), "error.code").String(); got != "previous_response_not_found" {
		t.Fatalf("turn 2 error.code = %q, want previous_response_not_found", got)
	}
	// The incomplete response never anchored a chain snapshot, so the second
	// turn must not even reach the upstream with an orphaned tool output.
	if len(*requests) != 1 {
		t.Fatalf("upstream requests = %d, want 1", len(*requests))
	}
}

func TestCodexExecutorExecuteStreamBootstrapBufferingRecordsChainSnapshot(t *testing.T) {
	executor, auth, requests := newCodexChainScriptedFixture(t, nil, codexBufferingConfig(true))

	for _, payload := range []string{codexChainTurn1Request, codexChainTurn2Request} {
		result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
			Model:   "gpt-5.4",
			Payload: []byte(payload),
		}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: true})
		if err != nil {
			t.Fatalf("ExecuteStream error: %v", err)
		}
		for range result.Chunks {
		}
	}

	assertCodexChainRebuiltSecondRequest(t, requests)
}

func TestCodexExecutorCountTokensUsesRebuiltChainInput(t *testing.T) {
	executor, auth, _ := newCodexChainFixture(t)

	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(codexChainTurn1Request),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false})
	if err != nil {
		t.Fatalf("turn 1 Execute error: %v", err)
	}

	countRebuilt, err := executor.CountTokens(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(codexChainTurn2Request),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response")})
	if err != nil {
		t.Fatalf("CountTokens error: %v", err)
	}

	helps.ClearCodexResponsesChainCache()
	countIncrementalOnly, err := executor.CountTokens(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(codexChainTurn2Request),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response")})
	if err != nil {
		t.Fatalf("CountTokens (no cache) error: %v", err)
	}

	rebuilt := gjson.GetBytes(countRebuilt.Payload, "response.usage.input_tokens").Int()
	incrementalOnly := gjson.GetBytes(countIncrementalOnly.Payload, "response.usage.input_tokens").Int()
	if rebuilt <= incrementalOnly {
		t.Fatalf("rebuilt token count = %d, want greater than incremental-only %d", rebuilt, incrementalOnly)
	}

	// The rebuilt input must tokenize exactly like the explicit full history.
	fullHistory := `{"model":"gpt-5.4","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in SF?"}]},{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{}"},{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`
	countControl, err := executor.CountTokens(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: []byte(fullHistory),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response")})
	if err != nil {
		t.Fatalf("CountTokens (control) error: %v", err)
	}
	control := gjson.GetBytes(countControl.Payload, "response.usage.input_tokens").Int()
	if rebuilt != control {
		t.Fatalf("rebuilt token count = %d, want exactly the full-history control count %d", rebuilt, control)
	}
}
