package helps

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tidwall/gjson"
)

func resetCodexResponsesChainForTest(t *testing.T) string {
	t.Helper()
	ClearCodexResponsesChainCache()
	originalDir := codexResponsesChainDir
	dir := t.TempDir()
	codexResponsesChainDir = dir
	t.Cleanup(func() {
		codexResponsesChainDir = originalDir
		ClearCodexResponsesChainCache()
	})
	return dir
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

// wipeCodexResponsesChainMemoryForTest simulates a process restart by
// dropping only the in-memory tier; the on-disk store survives.
func wipeCodexResponsesChainMemoryForTest() {
	codexResponsesChainMu.Lock()
	codexResponsesChainEntries = make(map[string]*codexResponsesChainEntry)
	codexResponsesChainTotalBytes = 0
	codexResponsesChainMu.Unlock()
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

func TestCodexResponsesChainLayeredTTLMemoryThenDiskThenExpiry(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	advance := withCodexResponsesChainClock(t, base)

	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(codexResponsesChainCompleted))
	incremental := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)

	// Within the memory TTL the hot cache serves the rebuild and refreshes.
	advance(CodexResponsesChainCacheTTL - time.Second)
	if updated := RepairCodexResponsesChainInput(incremental); bytes.Equal(updated, incremental) {
		t.Fatalf("expected rebuild from memory")
	}

	// Once the memory entry expires, the on-disk tier (24h TTL) still serves.
	wipeCodexResponsesChainMemoryForTest()
	advance(CodexResponsesChainDiskTTL - CodexResponsesChainCacheTTL)
	if updated := RepairCodexResponsesChainInput(incremental); bytes.Equal(updated, incremental) {
		t.Fatalf("expected rebuild from disk after memory expiry")
	}

	// Past the disk TTL the snapshot is gone for good.
	wipeCodexResponsesChainMemoryForTest()
	advance(2 * time.Minute)
	if updated := RepairCodexResponsesChainInput(incremental); !bytes.Equal(updated, incremental) {
		t.Fatalf("expected miss after disk TTL expiry, got %s", updated)
	}
}

func TestRepairCodexResponsesChainResultReportsMissingSnapshot(t *testing.T) {
	resetCodexResponsesChainForTest(t)

	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_gone","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	updated, parent, missing := RepairCodexResponsesChainInputWithResult(body)
	if missing != "resp_gone" {
		t.Fatalf("missing = %q, want resp_gone", missing)
	}
	if parent != "" {
		t.Fatalf("parent = %q, want empty on miss", parent)
	}
	if !bytes.Equal(updated, body) {
		t.Fatalf("body changed on miss: %s", updated)
	}

	// A hit reports the consumed parent instead.
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(codexResponsesChainCompleted))
	hitBody := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	updated, parent, missing = RepairCodexResponsesChainInputWithResult(hitBody)
	if missing != "" {
		t.Fatalf("missing = %q, want empty on hit", missing)
	}
	if parent != "resp_1" {
		t.Fatalf("parent = %q, want resp_1", parent)
	}
	if bytes.Equal(updated, hitBody) {
		t.Fatalf("expected rebuild on hit")
	}
}

func TestCodexResponsesChainReadsLegacyGzipFiles(t *testing.T) {
	dir := resetCodexResponsesChainForTest(t)
	// Hand-write a legacy gzip file instead of recording, simulating a
	// snapshot written before the zstd switch.
	record := codexResponsesChainDiskRecord{
		Input:      codexResponsesChainItemsFromResult(gjson.ParseBytes([]byte(codexResponsesChainUpstreamBody)).Get("input")),
		Output:     [][]byte{[]byte(`{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{}"}`)},
		RecordedAt: time.Now().Unix(),
	}
	payload, errMarshal := json.Marshal(record)
	if errMarshal != nil {
		t.Fatalf("marshal: %v", errMarshal)
	}
	legacyPath := filepath.Join(dir, "resp_1"+codexResponsesChainDiskLegacyExt)
	if errWrite := os.WriteFile(legacyPath, codexResponsesChainGzipCompress(payload), 0o600); errWrite != nil {
		t.Fatalf("write legacy file: %v", errWrite)
	}

	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	updated := RepairCodexResponsesChainInput(body)
	items := gjson.GetBytes(updated, "input").Array()
	if len(items) != 3 {
		t.Fatalf("input items = %d, want 3 from legacy gzip snapshot: %s", len(items), updated)
	}
}

func TestCodexResponsesChainSurvivesMemoryResetViaDisk(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(codexResponsesChainCompleted))

	// Simulate a process restart: memory tier is empty, disk tier survives.
	wipeCodexResponsesChainMemoryForTest()

	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	updated := RepairCodexResponsesChainInput(body)
	items := gjson.GetBytes(updated, "input").Array()
	if len(items) != 3 {
		t.Fatalf("input items = %d, want 3 after disk recovery: %s", len(items), updated)
	}
	// The promotion re-populates the memory tier.
	codexResponsesChainMu.Lock()
	_, promoted := codexResponsesChainEntries["resp_1"]
	codexResponsesChainMu.Unlock()
	if !promoted {
		t.Fatalf("expected disk hit to promote the entry into memory")
	}
}

func TestCodexResponsesChainFoldDeletesParentOnDiskAndMemory(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(codexResponsesChainCompleted))

	turn2Upstream := []byte(`{"model":"gpt-5.4","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in SF?"}]},{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{}"},{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	turn2Completed := `{"type":"response.completed","response":{"id":"resp_2","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"Sunny."}]}]}}`
	RecordCodexResponsesChainSnapshot(turn2Upstream, []byte(turn2Completed), "resp_1")

	if _, errStat := os.Stat(filepath.Join(codexResponsesChainDir, "resp_1"+codexResponsesChainDiskExt)); !os.IsNotExist(errStat) {
		t.Fatalf("parent disk file should be folded away: %v", errStat)
	}
	if _, errStat := os.Stat(filepath.Join(codexResponsesChainDir, "resp_2"+codexResponsesChainDiskExt)); errStat != nil {
		t.Fatalf("child disk file should exist: %v", errStat)
	}
	codexResponsesChainMu.Lock()
	_, parentInMemory := codexResponsesChainEntries["resp_1"]
	_, childInMemory := codexResponsesChainEntries["resp_2"]
	codexResponsesChainMu.Unlock()
	if parentInMemory {
		t.Fatalf("parent memory entry should be folded away")
	}
	if !childInMemory {
		t.Fatalf("child memory entry should exist")
	}

	// Repair with the folded parent id must now miss (upstream requests
	// referencing dead ancestors behave exactly like before the fix).
	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	if updated := RepairCodexResponsesChainInput(body); !bytes.Equal(updated, body) {
		t.Fatalf("body changed for folded parent: %s", updated)
	}
}

func TestCodexResponsesChainDiskFileIsCompressed(t *testing.T) {
	dir := resetCodexResponsesChainForTest(t)
	bigText := strings.Repeat("compressible storyboard workflow notes. ", 50_000)
	upstreamBody := []byte(`{"model":"gpt-5.4","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"` + bigText + `"}]}]}`)
	completed := []byte(`{"type":"response.completed","response":{"id":"resp_big","output":[{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"ok"}]}]}}`)
	RecordCodexResponsesChainSnapshot(upstreamBody, completed)

	info, errStat := os.Stat(filepath.Join(dir, "resp_big"+codexResponsesChainDiskExt))
	if errStat != nil {
		t.Fatalf("compressed snapshot file missing: %v", errStat)
	}
	rawBytes := int64(len(upstreamBody) + len(completed))
	if info.Size() >= rawBytes/4 {
		t.Fatalf("compressed file size = %d, want < 25%% of raw %d", info.Size(), rawBytes)
	}

	// The compressed file still round-trips through a rebuild.
	wipeCodexResponsesChainMemoryForTest()
	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_big","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"again"}]}]}`)
	updated := RepairCodexResponsesChainInput(body)
	items := gjson.GetBytes(updated, "input").Array()
	if len(items) != 3 {
		t.Fatalf("input items = %d, want 3 (cached user, cached output, new user) after compressed disk recovery", len(items))
	}
	if len(items[0].Get("content.0.text").String()) != len(bigText) {
		t.Fatalf("recovered user message text length mismatch")
	}
}

func TestCodexResponsesChainDiskCorruptFileIgnored(t *testing.T) {
	dir := resetCodexResponsesChainForTest(t)
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(codexResponsesChainCompleted))
	wipeCodexResponsesChainMemoryForTest()

	path := filepath.Join(dir, "resp_1"+codexResponsesChainDiskExt)
	if errWrite := os.WriteFile(path, []byte("not gzip at all"), 0o600); errWrite != nil {
		t.Fatalf("write corrupt file: %v", errWrite)
	}

	body := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_1","input":[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]}`)
	if updated := RepairCodexResponsesChainInput(body); !bytes.Equal(updated, body) {
		t.Fatalf("body changed for corrupt disk file: %s", updated)
	}
	if _, errStat := os.Stat(path); !os.IsNotExist(errStat) {
		t.Fatalf("corrupt file should be removed, stat err = %v", errStat)
	}
}

func TestCodexResponsesChainDiskSweepRemovesExpiredFiles(t *testing.T) {
	resetCodexResponsesChainForTest(t)
	RecordCodexResponsesChainSnapshot([]byte(codexResponsesChainUpstreamBody), []byte(codexResponsesChainCompleted))
	path := filepath.Join(codexResponsesChainDir, "resp_1"+codexResponsesChainDiskExt)
	expired := codexResponsesChainNow().Add(-CodexResponsesChainDiskTTL - time.Minute)
	if errTouch := os.Chtimes(path, expired, expired); errTouch != nil {
		t.Fatalf("chtimes: %v", errTouch)
	}

	sweepCodexResponsesChainDisk(codexResponsesChainNow())
	if _, errStat := os.Stat(path); !os.IsNotExist(errStat) {
		t.Fatalf("expired disk file should be swept, stat err = %v", errStat)
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
	if _, errStat := os.Stat(filepath.Join(codexResponsesChainDir, "resp_big"+codexResponsesChainDiskExt)); !os.IsNotExist(errStat) {
		t.Fatalf("oversized snapshot should not be written to disk")
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
	total := CodexResponsesChainCacheMaxEntries + 8
	for i := 0; i < total; i++ {
		completed := fmt.Sprintf(`{"type":"response.completed","response":{"id":"resp_%d","output":[{"type":"message","id":"msg_%d","content":[{"type":"output_text","text":"ok"}]}]}}`, i, i)
		RecordCodexResponsesChainSnapshot(upstreamBody, []byte(completed))
		advance(time.Millisecond)
	}

	first := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_0","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
	// resp_0 is evicted from memory but its disk file survives; wipe memory
	// is not needed - assert only the in-memory bound here.
	codexResponsesChainMu.Lock()
	entries := len(codexResponsesChainEntries)
	codexResponsesChainMu.Unlock()
	if entries > CodexResponsesChainCacheMaxEntries {
		t.Fatalf("entries = %d, want <= %d", entries, CodexResponsesChainCacheMaxEntries)
	}
	_ = first
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
