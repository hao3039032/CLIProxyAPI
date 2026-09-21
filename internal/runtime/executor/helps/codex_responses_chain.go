package helps

import (
	"bytes"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	// CodexResponsesChainCacheTTL bounds how long a completed response stays
	// eligible for previous_response_id reconstruction. Access refreshes the
	// entry so active conversations survive long tool loops.
	CodexResponsesChainCacheTTL = 1 * time.Hour

	// CodexResponsesChainCacheMaxEntries bounds the number of stored snapshots.
	CodexResponsesChainCacheMaxEntries = 4096

	// CodexResponsesChainCacheMaxBytesPerEntry skips oversized conversations
	// instead of displacing many healthy entries to fit a single huge one.
	CodexResponsesChainCacheMaxBytesPerEntry = 4 << 20 // 4MB

	// CodexResponsesChainCacheMaxTotalBytes bounds overall cache memory.
	CodexResponsesChainCacheMaxTotalBytes = 128 << 20 // 128MB

	// codexResponsesChainPurgeInterval throttles the O(n) expiry purge on the
	// record hot path; expired entries are still rejected on lookup, so this
	// only delays memory reclaim.
	codexResponsesChainPurgeInterval = CodexResponsesChainCacheTTL / 10

	// codexResponsesChainCacheEvictBatchSize leaves headroom after capacity so
	// bursts of completions do not trigger an eviction scan on every record.
	codexResponsesChainCacheEvictBatchSize = 128
)

// codexResponsesChainNow is injectable so TTL behaviour is testable without
// sleeping. It is swapped without synchronization: chain tests must not use
// t.Parallel() while overriding it (and parallel tests in this package must
// not exercise the chain cache).
var codexResponsesChainNow = time.Now

var codexResponsesChainLastPurge time.Time

type codexResponsesChainEntry struct {
	inputItems  [][]byte
	outputItems [][]byte
	bytes       int
	lastSeen    time.Time
}

var (
	codexResponsesChainMu         sync.Mutex
	codexResponsesChainEntries    = make(map[string]*codexResponsesChainEntry)
	codexResponsesChainTotalBytes int
)

// RecordCodexResponsesChainSnapshot stores the conversation snapshot of a
// completed Codex response so a later incremental request referencing it via
// previous_response_id can be rebuilt into a full stateless transcript.
// upstreamBody is the request body the upstream accepted; completedData is the
// native response.completed payload carrying response.id and response.output.
func RecordCodexResponsesChainSnapshot(upstreamBody, completedData []byte) {
	// Clone: the no-copy gjson result aliases completedData's backing buffer,
	// and a map key must not pin the whole response body in memory.
	respID := strings.Clone(strings.TrimSpace(util.GetGJSONBytesNoCopy(completedData, "response.id").String()))
	if respID == "" || len(upstreamBody) == 0 || len(completedData) == 0 {
		return
	}
	output := util.GetGJSONBytesNoCopy(completedData, "response.output")
	if !output.Exists() || !output.IsArray() {
		return
	}
	input := util.GetGJSONBytesNoCopy(upstreamBody, "input")
	if !input.Exists() || !input.IsArray() {
		return
	}
	inputItems := codexResponsesChainItemsFromResult(input)
	outputItems := codexResponsesChainItemsFromResult(output)
	if len(inputItems) == 0 && len(outputItems) == 0 {
		return
	}
	entryBytes := 0
	for _, item := range inputItems {
		entryBytes += len(item)
	}
	for _, item := range outputItems {
		entryBytes += len(item)
	}
	if entryBytes > CodexResponsesChainCacheMaxBytesPerEntry {
		return
	}

	now := codexResponsesChainNow()
	codexResponsesChainMu.Lock()
	defer codexResponsesChainMu.Unlock()
	if now.Sub(codexResponsesChainLastPurge) >= codexResponsesChainPurgeInterval {
		purgeExpiredCodexResponsesChainLocked(now)
		codexResponsesChainLastPurge = now
	}
	if existing, ok := codexResponsesChainEntries[respID]; ok {
		codexResponsesChainTotalBytes -= existing.bytes
	}
	codexResponsesChainEntries[respID] = &codexResponsesChainEntry{
		inputItems:  inputItems,
		outputItems: outputItems,
		bytes:       entryBytes,
		lastSeen:    now,
	}
	codexResponsesChainTotalBytes += entryBytes
	for len(codexResponsesChainEntries) > CodexResponsesChainCacheMaxEntries || codexResponsesChainTotalBytes > CodexResponsesChainCacheMaxTotalBytes {
		if evictOldestCodexResponsesChainLocked(codexResponsesChainCacheEvictBatchSize) == 0 {
			break
		}
	}
}

// RepairCodexResponsesChainInput rebuilds the full conversation input for an
// incremental Responses request that relies on server-side state through
// previous_response_id. The Codex HTTP upstream is stateless (requests are
// forwarded with store:false), so the cached snapshot of the referenced
// response is replayed in front of the new items, giving the upstream the
// exact shape a full-history client would send. Bodies without a string
// previous_response_id, cache misses, and inputs that already replay the
// cached items are returned unchanged. The previous_response_id field itself
// is left in place for the caller to remove.
func RepairCodexResponsesChainInput(body []byte) []byte {
	if len(body) == 0 {
		return body
	}
	previous := gjson.GetBytes(body, "previous_response_id")
	if previous.Type != gjson.String {
		return body
	}
	respID := strings.TrimSpace(previous.String())
	if respID == "" {
		return body
	}
	entry, ok := lookupCodexResponsesChainEntry(respID)
	if !ok {
		return body
	}
	input := util.GetGJSONBytesNoCopy(body, "input")
	if !input.Exists() || !input.IsArray() {
		return body
	}
	items := input.Array()
	if codexResponsesChainInputAlreadyReplayed(entry.outputItems, items) {
		return body
	}
	rebuilt := make([][]byte, 0, len(entry.inputItems)+len(entry.outputItems)+len(items))
	rebuilt = append(rebuilt, entry.inputItems...)
	rebuilt = append(rebuilt, entry.outputItems...)
	for _, item := range items {
		if len(item.Raw) == 0 {
			continue
		}
		rebuilt = append(rebuilt, []byte(item.Raw))
	}
	// Clients that echo output items back (e.g. reasoning items preserved
	// across store:false turns) re-send items already present in the cached
	// transcript; keep only the first occurrence of each id.
	rebuilt = dedupeCodexResponsesChainItemsByID(rebuilt)
	updated, errSet := sjson.SetRawBytes(body, "input", codexResponsesChainJoinItems(rebuilt))
	if errSet != nil {
		return body
	}
	return updated
}

// ClearCodexResponsesChainCache resets all stored conversation snapshots.
func ClearCodexResponsesChainCache() {
	codexResponsesChainMu.Lock()
	codexResponsesChainEntries = make(map[string]*codexResponsesChainEntry)
	codexResponsesChainTotalBytes = 0
	codexResponsesChainLastPurge = time.Time{}
	codexResponsesChainMu.Unlock()
}

func lookupCodexResponsesChainEntry(respID string) (codexResponsesChainEntry, bool) {
	now := codexResponsesChainNow()
	codexResponsesChainMu.Lock()
	defer codexResponsesChainMu.Unlock()
	entry, ok := codexResponsesChainEntries[respID]
	if !ok || entry == nil {
		return codexResponsesChainEntry{}, false
	}
	if now.Sub(entry.lastSeen) > CodexResponsesChainCacheTTL {
		codexResponsesChainTotalBytes -= entry.bytes
		delete(codexResponsesChainEntries, respID)
		return codexResponsesChainEntry{}, false
	}
	entry.lastSeen = now
	return codexResponsesChainEntry{
		inputItems:  cloneCodexResponsesChainItems(entry.inputItems),
		outputItems: cloneCodexResponsesChainItems(entry.outputItems),
	}, true
}

func codexResponsesChainItemsFromResult(result gjson.Result) [][]byte {
	items := result.Array()
	out := make([][]byte, 0, len(items))
	for _, item := range items {
		if len(item.Raw) == 0 {
			continue
		}
		out = append(out, []byte(item.Raw))
	}
	return out
}

func cloneCodexResponsesChainItems(items [][]byte) [][]byte {
	cloned := make([][]byte, 0, len(items))
	for _, item := range items {
		cloned = append(cloned, append([]byte(nil), item...))
	}
	return cloned
}

func codexResponsesChainJoinItems(items [][]byte) []byte {
	var buf bytes.Buffer
	buf.WriteByte('[')
	for i, item := range items {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.Write(item)
	}
	buf.WriteByte(']')
	return buf.Bytes()
}

// dedupeCodexResponsesChainItemsByID keeps only the first occurrence of each
// item carrying an id; items without an id are always kept.
func dedupeCodexResponsesChainItemsByID(items [][]byte) [][]byte {
	seen := make(map[string]struct{})
	deduped := make([][]byte, 0, len(items))
	for _, item := range items {
		itemID := strings.TrimSpace(gjson.GetBytes(item, "id").String())
		if itemID != "" {
			if _, ok := seen[itemID]; ok {
				continue
			}
			seen[itemID] = struct{}{}
		}
		deduped = append(deduped, item)
	}
	return deduped
}

// codexResponsesChainInputAlreadyReplayed reports whether the request input
// already contains the cached output items, meaning the client replayed the
// full history itself and reconstruction would duplicate the transcript.
// Tool calls are matched by call_id; message items are matched by id.
// Reasoning items are excluded: clients echo them back alongside incremental
// input to preserve reasoning across store:false turns, and that alone must
// not be mistaken for a full replay.
func codexResponsesChainInputAlreadyReplayed(outputItems [][]byte, items []gjson.Result) bool {
	if len(outputItems) == 0 {
		return false
	}
	cachedCallIDs := make(map[string]struct{})
	cachedItemIDs := make(map[string]struct{})
	for _, raw := range outputItems {
		itemType := strings.TrimSpace(gjson.GetBytes(raw, "type").String())
		if itemType == "function_call" || itemType == "custom_tool_call" {
			if callID := strings.TrimSpace(gjson.GetBytes(raw, "call_id").String()); callID != "" {
				cachedCallIDs[callID] = struct{}{}
			}
			continue
		}
		if itemType == "reasoning" {
			continue
		}
		if itemID := strings.TrimSpace(gjson.GetBytes(raw, "id").String()); itemID != "" {
			cachedItemIDs[itemID] = struct{}{}
		}
	}
	if len(cachedCallIDs) == 0 && len(cachedItemIDs) == 0 {
		return false
	}
	for _, item := range items {
		itemType := strings.TrimSpace(item.Get("type").String())
		if itemType == "function_call" || itemType == "custom_tool_call" {
			if _, ok := cachedCallIDs[strings.TrimSpace(item.Get("call_id").String())]; ok {
				return true
			}
		}
		if itemID := strings.TrimSpace(item.Get("id").String()); itemID != "" {
			if _, ok := cachedItemIDs[itemID]; ok {
				return true
			}
		}
	}
	return false
}

func purgeExpiredCodexResponsesChainLocked(now time.Time) {
	for respID, entry := range codexResponsesChainEntries {
		if entry == nil {
			delete(codexResponsesChainEntries, respID)
			continue
		}
		if now.Sub(entry.lastSeen) > CodexResponsesChainCacheTTL {
			codexResponsesChainTotalBytes -= entry.bytes
			delete(codexResponsesChainEntries, respID)
		}
	}
}

func evictOldestCodexResponsesChainLocked(count int) int {
	if count <= 0 || len(codexResponsesChainEntries) == 0 {
		return 0
	}
	type candidate struct {
		respID   string
		lastSeen time.Time
	}
	candidates := make([]candidate, 0, len(codexResponsesChainEntries))
	for respID, entry := range codexResponsesChainEntries {
		candidates = append(candidates, candidate{respID: respID, lastSeen: entry.lastSeen})
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].lastSeen.Before(candidates[j].lastSeen)
	})
	if count > len(candidates) {
		count = len(candidates)
	}
	for i := 0; i < count; i++ {
		if entry, ok := codexResponsesChainEntries[candidates[i].respID]; ok {
			codexResponsesChainTotalBytes -= entry.bytes
			delete(codexResponsesChainEntries, candidates[i].respID)
		}
	}
	return count
}
