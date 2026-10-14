package executor

import (
	"fmt"
	"os"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor/helps"
)

// TestMain redirects the Codex responses chain snapshot store to a temporary
// directory so executor tests that complete Codex responses do not write
// responses-chain-cache/ into the source tree.
func TestMain(m *testing.M) {
	dir, errTemp := os.MkdirTemp("", "executor-responses-chain-cache-")
	if errTemp != nil {
		fmt.Fprintf(os.Stderr, "create responses chain cache temp dir: %v\n", errTemp)
		os.Exit(1)
	}
	restore := helps.SetCodexResponsesChainDirForTest(dir)
	code := m.Run()
	restore()
	if errRemove := os.RemoveAll(dir); errRemove != nil {
		fmt.Fprintf(os.Stderr, "remove responses chain cache temp dir: %v\n", errRemove)
	}
	os.Exit(code)
}
