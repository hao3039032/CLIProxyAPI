package helps

import (
	"bytes"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tidwall/gjson"
)

func resetCodexResponsesChainForTest(t *testing.T) {
	t.Helper()
	ClearCodexResponsesChainCache()
	t.Cleanup(ClearCodexResponsesChainCache)
}

func withCodexResponsesChainClock(t *testing.T, start time.Time) func(advance time.Duration) {
	t.Helper()
	current := start
	original := codexResponsesChainNow
	codexResponsesChainNow = func() time.Time { return current }
	advance := func(d time.Duration) { current = current.Add(d) }
	t.Cleanup(func() { codexResponsesChainNow = original })
	return advance
}

const codexResponsesChainUpstreamBody = `{"model":"gpt-5.4","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in SF?"}]}]}`

const codexResponsesChainCompleted = `{"type":"response.completed","response":{"id":"resp_1","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{}"}]}}`

func TestRepairCodexResponsesChainInputRebuildsIncrementalRequest(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(codexResponsesChainCompleted))

	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	updated := RepairCodexResponsesChainInput(body)

	items := gjson.GetBytes(updated, "input").Array()
	if len(items) != 3 {
		t.Fatalf("input items = %d, want 3: %s", len(items), updated)
	}
	if got := items[0].Get("type").String(); got != "message" {
		t.Fatalf("items[0].type = %q, want message", got)
	}
	if got := items[1].Get("call_id").String(); got != "call_1" {
		t.Fatalf("items[1].call_id = %q, want call_1", got)
	}
	if got := items[1].Get("type").String(); got != "function_call" {
		t.Fatalf("items[1].type = %q, want function_call", got)
	}
	if got := items[2].Get("type").String(); got != "function_call_output" {
		t.Fatalf("items[2].type = %q, want function_call_output", got)
	}
	// The field itself stays for the caller to remove.
	if got := gjson.GetBytes(updated, "previous_response_id").String(); got != "resp_1" {
		t.Fatalf("previous_response_id = %q, want resp_1", got)
	}
}

func TestRepairCodexResponsesChainInputRebuildsNewUserMessage(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	completed := `{"type":"response.completed","response":{"id":"resp_2","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"It is sunny."}]}]}}`
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(completed))

	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_2","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"thanks"}]}]}`)
	updated := RepairCodexResponsesChainInput(body)

	items := gjson.GetBytes(updated, "input").Array()
	if len(items) != 3 {
		t.Fatalf("input items = %d, want 3: %s", len(items), updated)
	}
	if got := items[1].Get("id").String(); got != "msg_1" {
		t.Fatalf("items[1].id = %q, want msg_1", got)
	}
	if got := items[2].Get("content.0.text").String(); got != "thanks" {
		t.Fatalf("items[2].text = %q, want thanks", got)
	}
}

func TestRepairCodexResponsesChainInputWithoutPreviousResponseID(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(codexResponsesChainCompleted))

	body := []byte(`{"model":"gpt-5.4","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	if updated := RepairCodexResponsesChainInput(body); !bytes.Equal(updated, body) {
		t.Fatalf("body changed without previous_response_id: %s", updated)
	}
}

func TestRepairCodexResponsesChainInputCacheMiss(t *testing.T) {
	resetCodexResponsesChainForTest(t)

	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_unknown","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	if updated := RepairCodexResponsesChainInput(body); !bytes.Equal(updated, body) {
		t.Fatalf("body changed on cache miss: %s", updated)
	}
}

func TestRepairCodexResponsesChainInputSkipsFullReplay(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(codexResponsesChainCompleted))

	callReplay := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in SF?"}]},{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{}"},{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	if updated := RepairCodexResponsesChainInput(callReplay); !bytes.Equal(updated, callReplay) {
		t.Fatalf("body changed although client replayed the call: %s", updated)
	}

	messageCompleted := `{"type":"response.completed","response":{"id":"resp_msg","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"It is sunny."}]}]}}`
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(messageCompleted))
	messageReplay := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_msg","input":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"It is sunny."}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"thanks"}]}]}`)
	if updated := RepairCodexResponsesChainInput(messageReplay); !bytes.Equal(updated, messageReplay) {
		t.Fatalf("body changed although client replayed the message item: %s", updated)
	}
}

func TestRepairCodexResponsesChainInputSkipsFullReplayForCustomToolCall(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	completed := `{"type":"response.completed","response":{"id":"resp_custom","output":[{"type":"custom_tool_call","id":"ctc_1","call_id":"call_c1","name":"apply_patch","input":"patch"}]}}`
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(completed))

	replay := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_custom","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in SF?"}]},{"type":"custom_tool_call","id":"ctc_1","call_id":"call_c1","name":"apply_patch","input":"patch"},{"type":"custom_tool_call_output","id":"ctco_1","call_id":"call_c1","output":"done"}]}`)
	if updated := RepairCodexResponsesChainInput(replay); !bytes.Equal(updated, replay) {
		t.Fatalf("body changed although client replayed the custom tool call: %s", updated)
	}
}

func TestRepairCodexResponsesChainInputRebuildsWithEchoedReasoningAndDedupes(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	completed := `{"type":"response.completed","response":{"id":"resp_reason","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"enc"},{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{}"}]}}`
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(completed))

	// The client echoes the reasoning item back alongside the incremental
	// tool output; the rebuild must proceed and drop the duplicate id.
	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_reason","input":[{"type":"reasoning","id":"rs_1","encrypted_content":"enc"},{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	updated := RepairCodexResponsesChainInput(body)

	items := gjson.GetBytes(updated, "input").Array()
	if len(items) != 4 {
		t.Fatalf("input items = %d, want 4 (user, reasoning, function_call, function_call_output): %s", len(items), updated)
	}
	if got := items[1].Get("id").String(); got != "rs_1" {
		t.Fatalf("items[1].id = %q, want rs_1", got)
	}
	reasoningCount := 0
	for _, item := range items {
		if item.Get("type").String() == "reasoning" {
			reasoningCount++
		}
	}
	if reasoningCount != 1 {
		t.Fatalf("reasoning items = %d, want 1 after dedup: %s", reasoningCount, updated)
	}
}

func TestCodexResponsesChainCacheTTLExpiryAndRefresh(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	advance := withCodexResponsesChainClock(t, base)

	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(codexResponsesChainCompleted))
	advance(CodexResponsesChainCacheTTL - time.Second)

	incremental := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	if updated := RepairCodexResponsesChainInput(incremental); bytes.Equal(updated, incremental) {
		t.Fatalf("expected rebuild before TTL expiry")
	}

	// The successful lookup refreshed lastSeen; the entry survives an hour
	// measured from the refresh point, then expires.
	advance(CodexResponsesChainCacheTTL - time.Second)
	if updated := RepairCodexResponsesChainInput(incremental); bytes.Equal(updated, incremental) {
		t.Fatalf("expected rebuild after refresh")
	}
	advance(CodexResponsesChainCacheTTL + time.Second)
	if updated := RepairCodexResponsesChainInput(incremental); !bytes.Equal(updated, incremental) {
		t.Fatalf("expected miss after TTL expiry, got %s", updated)
	}
}

func TestRecordCodexResponsesChainSnapshotSkipsOversizedEntry(t *testing.T) {
	resetCodexResponsesChainForTest(t)

	hugeOutput := fmt.Sprintf(`{"type":"response.completed","response":{"id":"resp_big","output":[{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"%s"}]}]}}`, strings.Repeat("a", CodexResponsesChainCacheMaxBytesPerEntry))
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(hugeOutput))

	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_big","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
	if updated := RepairCodexResponsesChainInput(body); !bytes.Equal(updated, body) {
		t.Fatalf("oversized snapshot was recorded: %s", updated)
	}
}

func TestRecordCodexResponsesChainSnapshotSkipsInvalidPayloads(t *testing.T) {
	resetCodexResponsesChainForTest(t)

	cases := []struct {
		name          string
		upstreamBody  []byte
		completedData []byte
	}{
		{"missing response id", []byte(codexResponsesChainUpstreamBody), []byte(`{"type":"response.completed","response":{"output":[]}}`)},
		{"output not array", []byte(codexResponsesChainUpstreamBody), []byte(`{"type":"response.completed","response":{"id":"resp_1","output":null}}`)},
		{"output key absent", []byte(codexResponsesChainUpstreamBody), []byte(`{"type":"response.completed","response":{"id":"resp_1"}}`)},
		{"input not array", []byte(`{"model":"gpt-5.4","input":"hello"}`), []byte(codexResponsesChainCompleted)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			RecordCodexResponsesChainSnapshot(tc.upstreamBody, tc.completedData)
			body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
			if updated := RepairCodexResponsesChainInput(body); !bytes.Equal(updated, body) {
				t.Fatalf("invalid payload was recorded: %s", updated)
			}
		})
	}
}

func TestCodexResponsesChainCacheEvictsBeyondMaxEntries(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	advance := withCodexResponsesChainClock(t, base)

	upstreamBody := []byte(`{"model":"gpt-5.4","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
	total := CodexResponsesChainCacheMaxEntries + codexResponsesChainCacheEvictBatchSize + 8
	for i := 0; i < total; i++ {
		completed := fmt.Sprintf(`{"type":"response.completed","response":{"id":"resp_%d","output":[{"type":"message","id":"msg_%d","content":[{"type":"output_text","text":"ok"}]}]}}`, i, i)
		RecordCodexResponsesChainSnapshot(upstreamBody, []byte(completed))
		advance(time.Millisecond)
	}

	first := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_0","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
	if updated := RepairCodexResponsesChainInput(first); !bytes.Equal(updated, first) {
		t.Fatalf("oldest entry was not evicted")
	}

	last := []byte(fmt.Sprintf(`{"model":"gpt-5.4","previous_response_id":"resp_%d","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`, total-1))
	if updated := RepairCodexResponsesChainInput(last); bytes.Equal(updated, last) {
		t.Fatalf("newest entry was evicted")
	}

	codexResponsesChainMu.Lock()
	entries := len(codexResponsesChainEntries)
	codexResponsesChainMu.Unlock()
	if entries > CodexResponsesChainCacheMaxEntries {
		t.Fatalf("entries = %d, want <= %d", entries, CodexResponsesChainCacheMaxEntries)
	}
}

func TestCodexResponsesChainConcurrentRecordAndRepair(t *testing.T) {
	resetCodexResponsesChainForTest(t)

	upstreamBody := []byte(codexResponsesChainUpstreamBody)
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				completed := fmt.Sprintf(`{"type":"response.completed","response":{"id":"resp_%d_%d","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{}"}]}}`, worker, i)
				RecordCodexResponsesChainSnapshot(upstreamBody, []byte(completed))
				body := []byte(fmt.Sprintf(`{"model":"gpt-5.4","previous_response_id":"resp_%d_%d","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`, worker, i))
				RepairCodexResponsesChainInput(body)
			}
		}(worker)
	}
	wg.Wait()
}
