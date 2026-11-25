package fetch

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Import puts the local file src at dest after checking its sha256 (skipped when sha256 is
// empty). It hard-links when the filesystem allows, else copies through a temp file and renames.
// A dest that already holds the same content is left in place.
func Import(src, dest, sha256hex string) error {
	sha256hex = strings.ToLower(strings.TrimSpace(sha256hex))
	st, err := os.Stat(src)
	if err != nil {
		return err
	}
	if !st.Mode().IsRegular() {
		return fmt.Errorf("import: %s is not a regular file", src)
	}
	if sha256hex != "" {
		got, err := hashFile(context.Background(), src, nil)
		if err != nil {
			return err
		}
		if got != sha256hex {
			return fmt.Errorf("import %s: %w: got %s, want %s", src, ErrHashMismatch, got, sha256hex)
		}
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	if ds, err := os.Stat(dest); err == nil {
		if os.SameFile(st, ds) {
			return nil
		}
		if ds.Size() == st.Size() && sha256hex != "" {
			if got, err := hashFile(context.Background(), dest, nil); err == nil && got == sha256hex {
				return nil
			}
		}
	}
	tmp := dest + ".import"
	os.Remove(tmp)
	if err := os.Link(src, tmp); err != nil {
		in, err := os.Open(src)
		if err != nil {
			return err
		}
		defer in.Close()
		out, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0o644)
		if err != nil {
			return err
		}
		if _, err := io.Copy(out, in); err != nil {
			out.Close()
			os.Remove(tmp)
			return err
		}
		if err := out.Close(); err != nil {
			os.Remove(tmp)
			return err
		}
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}
