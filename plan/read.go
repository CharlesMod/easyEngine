package plan

import (
	"fmt"
	"os"

	"github.com/CharlesMod/easyEngine/core/gguf"
)

// ReadInfo opens a model file and returns its header's Info. The core's reader never opens a
// file: it is handed the open file and its size here.
func ReadInfo(path string) (gguf.Info, error) {
	fh, err := os.Open(path)
	if err != nil {
		return gguf.Info{}, err
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil {
		return gguf.Info{}, err
	}
	f, err := gguf.Parse(fh, st.Size())
	if err != nil {
		return gguf.Info{}, fmt.Errorf("%s: %w", path, err)
	}
	return f.Info()
}
