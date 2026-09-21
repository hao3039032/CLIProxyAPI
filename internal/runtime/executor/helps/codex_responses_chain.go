package helps

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	log "github.com/sirupsen/logrus"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	// CodexResponsesChainCacheTTL bounds how long a completed response stays
	// eligible for previous_response_id reconstruction in process memory.
	// Access refreshes the entry so active conversations survive long tool
	// loops.
	CodexResponsesChainCacheTTL = 1 * time.Hour

	// CodexResponsesChainDiskTTL bounds on-disk snapshot retention; it is
	// longer than the memory TTL so conversations that idle past a deploy or
	// a memory eviction can still be reconstructed.
	CodexResponsesChainDiskTTL = 24 * time.Hour

	// CodexResponsesChainCacheMaxEntries bounds the number of in-memory
	// snapshots; the global byte budget is the primary memory limiter.
	CodexResponsesChainCacheMaxEntries = 1024

	// CodexResponsesChainCacheMaxBytesPerEntry skips oversized conversations
	// instead of trying to store a single huge transcript. Must exceed the
	// largest accepted request: gateways in front of this proxy allow ~100MB
	// single requests, and image-heavy transcripts embed multi-MB base64
	// blocks per turn (observed ~10MB turn-1, ceiling ~100MB).
	CodexResponsesChainCacheMaxBytesPerEntry = 128 << 20 // 128MB

	// CodexResponsesChainCacheMaxTotalBytes bounds the in-memory hot cache.
	// The on-disk store is the authoritative tier; memory only accelerates
	// lookups.
	CodexResponsesChainCacheMaxTotalBytes = 512 << 20 // 512MB

	// CodexResponsesChainDiskMaxTotalBytes bounds total on-disk snapshot
	// bytes; the throttled sweep deletes the oldest files beyond it.
	CodexResponsesChainDiskMaxTotalBytes = 8 << 30 // 8GB

	// codexResponsesChainPurgeInterval throttles the O(n) expiry purge and
	// disk sweep; expired entries are still rejected on lookup, so this only
	// delays memory and disk reclaim.
	codexResponsesChainPurgeInterval = 6 * time.Minute

	// codexResponsesChainCacheDir is the on-disk snapshot store, relative to
	// the process working directory. Deployments that want snapshots to
	// survive container recreation should volume-mount it.
	codexResponsesChainCacheDir = "responses-chain-cache"

	// codexResponsesChainDiskExt is the compressed snapshot file extension.
	codexResponsesChainDiskExt = ".json.gz"
)

// codexResponsesChainNow is injectable so TTL behaviour is testable without
// sleeping. It is swapped without synchronization: chain tests must not use
// t.Parallel() while overriding it (and parallel tests in this package must
// not exercise the chain cache).
var codexResponsesChainNow = time.Now

// codexResponsesChainDir is injectable for tests.
var codexResponsesChainDir = codexResponsesChainCacheDir

var codexResponsesChainLastSweep time.Time

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

type codexResponsesChainDiskRecord struct {
	Input      [][]byte `json:"input"`
	Output     [][]byte `json:"output"`
	RecordedAt int64    `json:"recorded_at"`
}

// RecordCodexResponsesChainSnapshot stores the conversation snapshot of a
// completed Codex response so a later incremental request referencing it via
// previous_response_id can be rebuilt into a full stateless transcript.
// upstreamBody is the request body the upstream accepted; completedData is the
// native response.completed payload carrying response.id and response.output.
// A non-empty replacesRespID (the snapshot consumed while repairing this
// request) is folded away: once the successor is durably recorded the parent
// chain entry will never be referenced again.
func RecordCodexResponsesChainSnapshot(upstreamBody, completedData []byte, replacesRespID ...string) {
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
		log.Infof("codex responses chain: skip recording resp=%s entryBytes=%d over per-entry cap %d", respID, entryBytes, CodexResponsesChainCacheMaxBytesPerEntry)
		return
	}

	now := codexResponsesChainNow()
	writeCodexResponsesChainDisk(respID, inputItems, outputItems, now)

	foldedParent := ""
	if len(replacesRespID) > 0 {
		foldedParent = strings.Clone(strings.TrimSpace(replacesRespID[0]))
		if foldedParent == respID {
			foldedParent = ""
		}
	}

	codexResponsesChainMu.Lock()
	defer codexResponsesChainMu.Unlock()
	sweepCodexResponsesChainLocked(now)
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
	if foldedParent != "" {
		if parent, ok := codexResponsesChainEntries[foldedParent]; ok {
			codexResponsesChainTotalBytes -= parent.bytes
			delete(codexResponsesChainEntries, foldedParent)
		}
		deleteCodexResponsesChainDisk(foldedParent)
	}
	if foldedParent != "" {
		log.Infof("codex responses chain: recorded snapshot resp=%s inputItems=%d outputItems=%d bytes=%d folds=%s", respID, len(inputItems), len(outputItems), entryBytes, foldedParent)
	} else {
		log.Infof("codex responses chain: recorded snapshot resp=%s inputItems=%d outputItems=%d bytes=%d", respID, len(inputItems), len(outputItems), entryBytes)
	}
	for len(codexResponsesChainEntries) > CodexResponsesChainCacheMaxEntries || codexResponsesChainTotalBytes > CodexResponsesChainCacheMaxTotalBytes {
		if !evictOldestCodexResponsesChainLocked() {
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
	updated, _ := RepairCodexResponsesChainInputWithParent(body)
	return updated
}

// RepairCodexResponsesChainInputWithParent behaves like
// RepairCodexResponsesChainInput and additionally returns the resp id whose
// snapshot was consumed to rebuild the input, so the caller can fold it away
// once this turn's response completes.
func RepairCodexResponsesChainInputWithParent(body []byte) ([]byte, string) {
	if len(body) == 0 {
		return body, ""
	}
	previous := gjson.GetBytes(body, "previous_response_id")
	if previous.Type != gjson.String {
		return body, ""
	}
	respID := strings.TrimSpace(previous.String())
	if respID == "" {
		return body, ""
	}
	entry, ok := lookupCodexResponsesChainEntry(respID)
	if !ok {
		log.Infof("codex responses chain: cache miss for previous_response_id=%s", respID)
		return body, ""
	}
	input := util.GetGJSONBytesNoCopy(body, "input")
	if !input.Exists() || !input.IsArray() {
		return body, ""
	}
	items := input.Array()
	if codexResponsesChainInputAlreadyReplayed(entry.outputItems, items) {
		return body, ""
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
		return body, ""
	}
	log.Infof("codex responses chain: rebuilt input for previous_response_id=%s cachedInput=%d cachedOutput=%d newItems=%d rebuilt=%d", respID, len(entry.inputItems), len(entry.outputItems), len(items), len(rebuilt))
	return updated, respID
}

// ClearCodexResponsesChainCache resets all in-memory and on-disk snapshots.
func ClearCodexResponsesChainCache() {
	codexResponsesChainMu.Lock()
	codexResponsesChainEntries = make(map[string]*codexResponsesChainEntry)
	codexResponsesChainTotalBytes = 0
	codexResponsesChainLastSweep = time.Time{}
	codexResponsesChainMu.Unlock()
	if entries, errList := os.ReadDir(codexResponsesChainDir); errList == nil {
		for _, entry := range entries {
			if strings.HasSuffix(entry.Name(), codexResponsesChainDiskExt) {
				_ = os.Remove(filepath.Join(codexResponsesChainDir, entry.Name()))
			}
		}
	}
}

func lookupCodexResponsesChainEntry(respID string) (codexResponsesChainEntry, bool) {
	now := codexResponsesChainNow()
	codexResponsesChainMu.Lock()
	entry, ok := codexResponsesChainEntries[respID]
	if ok && entry != nil && now.Sub(entry.lastSeen) <= CodexResponsesChainCacheTTL {
		entry.lastSeen = now
		codexResponsesChainMu.Unlock()
		return codexResponsesChainEntry{
			inputItems:  cloneCodexResponsesChainItems(entry.inputItems),
			outputItems: cloneCodexResponsesChainItems(entry.outputItems),
		}, true
	}
	if ok {
		codexResponsesChainTotalBytes -= entry.bytes
		delete(codexResponsesChainEntries, respID)
	}
	codexResponsesChainMu.Unlock()

	// Memory miss: fall back to the on-disk store and promote it.
	if record, ok := readCodexResponsesChainDisk(respID, now); ok {
		entryBytes := 0
		for _, item := range record.Input {
			entryBytes += len(item)
		}
		for _, item := range record.Output {
			entryBytes += len(item)
		}
		promoteCodexResponsesChainEntry(respID, record.Input, record.Output, entryBytes, now)
		log.Infof("codex responses chain: disk hit for previous_response_id=%s", respID)
		return codexResponsesChainEntry{
			inputItems:  cloneCodexResponsesChainItems(record.Input),
			outputItems: cloneCodexResponsesChainItems(record.Output),
		}, true
	}
	return codexResponsesChainEntry{}, false
}

func promoteCodexResponsesChainEntry(respID string, inputItems, outputItems [][]byte, entryBytes int, now time.Time) {
	if entryBytes > CodexResponsesChainCacheMaxBytesPerEntry {
		return
	}
	codexResponsesChainMu.Lock()
	defer codexResponsesChainMu.Unlock()
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
		if !evictOldestCodexResponsesChainLocked() {
			break
		}
	}
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

// codexResponsesChainSafeFileName reports whether a resp id is safe to use as
// a cache file name stem.
func codexResponsesChainSafeFileName(respID string) bool {
	if len(respID) == 0 || len(respID) > 128 {
		return false
	}
	for _, r := range respID {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-', r == '.':
		default:
			return false
		}
	}
	return !strings.Contains(respID, "..")
}

func writeCodexResponsesChainDisk(respID string, inputItems, outputItems [][]byte, now time.Time) {
	if !codexResponsesChainSafeFileName(respID) {
		return
	}
	if errMkdir := os.MkdirAll(codexResponsesChainDir, 0o700); errMkdir != nil {
		return
	}
	record := codexResponsesChainDiskRecord{Input: inputItems, Output: outputItems, RecordedAt: now.Unix()}
	var compressed bytes.Buffer
	gzipWriter, errWriter := gzip.NewWriterLevel(&compressed, gzip.BestSpeed)
	if errWriter != nil {
		return
	}
	if errEncode := json.NewEncoder(gzipWriter).Encode(record); errEncode != nil {
		return
	}
	if errClose := gzipWriter.Close(); errClose != nil {
		return
	}
	path := filepath.Join(codexResponsesChainDir, respID+codexResponsesChainDiskExt)
	tmp := path + ".tmp"
	if errWrite := os.WriteFile(tmp, compressed.Bytes(), 0o600); errWrite != nil {
		return
	}
	if errRename := os.Rename(tmp, path); errRename != nil {
		_ = os.Remove(tmp)
		return
	}
}

func readCodexResponsesChainDisk(respID string, now time.Time) (codexResponsesChainDiskRecord, bool) {
	if !codexResponsesChainSafeFileName(respID) {
		return codexResponsesChainDiskRecord{}, false
	}
	path := filepath.Join(codexResponsesChainDir, respID+codexResponsesChainDiskExt)
	remove := func() { _ = os.Remove(path) }
	raw, errRead := os.ReadFile(path)
	if errRead != nil {
		return codexResponsesChainDiskRecord{}, false
	}
	gzipReader, errGzip := gzip.NewReader(bytes.NewReader(raw))
	if errGzip != nil {
		remove()
		return codexResponsesChainDiskRecord{}, false
	}
	payload, errAll := io.ReadAll(gzipReader)
	_ = gzipReader.Close()
	if errAll != nil {
		remove()
		return codexResponsesChainDiskRecord{}, false
	}
	var record codexResponsesChainDiskRecord
	if errUnmarshal := json.Unmarshal(payload, &record); errUnmarshal != nil {
		remove()
		return codexResponsesChainDiskRecord{}, false
	}
	if len(record.Input) == 0 && len(record.Output) == 0 {
		remove()
		return codexResponsesChainDiskRecord{}, false
	}
	if now.Sub(time.Unix(record.RecordedAt, 0)) > CodexResponsesChainDiskTTL {
		remove()
		return codexResponsesChainDiskRecord{}, false
	}
	return record, true
}

func deleteCodexResponsesChainDisk(respID string) {
	if !codexResponsesChainSafeFileName(respID) {
		return
	}
	_ = os.Remove(filepath.Join(codexResponsesChainDir, respID+codexResponsesChainDiskExt))
}

// sweepCodexResponsesChainLocked removes expired memory entries and, at a
// throttled cadence, sweeps expired and over-budget disk files.
func sweepCodexResponsesChainLocked(now time.Time) {
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
	if now.Sub(codexResponsesChainLastSweep) < codexResponsesChainPurgeInterval {
		return
	}
	codexResponsesChainLastSweep = now
	codexResponsesChainMu.Unlock()
	sweepCodexResponsesChainDisk(now)
	codexResponsesChainMu.Lock()
}

func sweepCodexResponsesChainDisk(now time.Time) {
	entries, errList := os.ReadDir(codexResponsesChainDir)
	if errList != nil {
		return
	}
	type diskCandidate struct {
		name    string
		modTime time.Time
		size    int64
	}
	candidates := make([]diskCandidate, 0, len(entries))
	var totalBytes int64
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, codexResponsesChainDiskExt) {
			continue
		}
		info, errInfo := entry.Info()
		if errInfo != nil {
			continue
		}
		if now.Sub(info.ModTime()) > CodexResponsesChainDiskTTL {
			_ = os.Remove(filepath.Join(codexResponsesChainDir, name))
			continue
		}
		totalBytes += info.Size()
		candidates = append(candidates, diskCandidate{name: name, modTime: info.ModTime(), size: info.Size()})
	}
	if totalBytes <= CodexResponsesChainDiskMaxTotalBytes {
		return
	}
	// Over budget: drop oldest files first until back under the cap.
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].modTime.Before(candidates[i].modTime) {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}
	for _, candidate := range candidates {
		if totalBytes <= CodexResponsesChainDiskMaxTotalBytes {
			break
		}
		_ = os.Remove(filepath.Join(codexResponsesChainDir, candidate.name))
		totalBytes -= candidate.size
	}
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

// evictOldestCodexResponsesChainLocked removes exactly the single
// least-recently-used entry and reports whether one was removed. Evicting
// one at a time keeps large image-heavy snapshots from wiping the whole cache
// when the global budget is exceeded. Only the in-memory copy is dropped;
// the authoritative disk file stays until its own TTL or sweep.
func evictOldestCodexResponsesChainLocked() bool {
	oldestID := ""
	var oldestSeen time.Time
	found := false
	for respID, entry := range codexResponsesChainEntries {
		if entry == nil {
			delete(codexResponsesChainEntries, respID)
			return true
		}
		if !found || entry.lastSeen.Before(oldestSeen) {
			oldestID = respID
			oldestSeen = entry.lastSeen
			found = true
		}
	}
	if !found {
		return false
	}
	codexResponsesChainTotalBytes -= codexResponsesChainEntries[oldestID].bytes
	delete(codexResponsesChainEntries, oldestID)
	return true
}
