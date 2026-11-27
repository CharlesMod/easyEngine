package pair

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// EngineIDFile holds the engine's stable id in the data dir.
const EngineIDFile = "engine-id"

// EngineID is this data dir's stable random id (32 hex characters), made on first use and kept
// in EngineIDFile. It names the engine to clients (hello, mDNS, the Allow answer) and says
// nothing else about the machine. Two processes making it at once agree: the file is put in
// place by a hard link that fails if one is there already.
func EngineID(dir string) (string, error) {
	path := filepath.Join(dir, EngineIDFile)
	if id, err := readID(path); err == nil {
		return id, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	id, err := randomID()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp(dir, EngineIDFile+".*")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.WriteString(id + "\n"); err != nil {
		tmp.Close()
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := os.Link(tmp.Name(), path); err != nil {
		if errors.Is(err, os.ErrExist) {
			return readID(path)
		}
		return "", err
	}
	return id, nil
}

func readID(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	id := strings.TrimSpace(string(b))
	if len(id) < 16 || strings.Trim(id, "0123456789abcdef") != "" {
		return "", fmt.Errorf("%s: not an engine id", path)
	}
	return id, nil
}
