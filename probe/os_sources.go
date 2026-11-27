package probe

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"

	"github.com/CharlesMod/easyEngine/probe/smiloop"
)

// OSSources reads the real machine.
type OSSources struct{}

func (OSSources) Platform() (string, string) { return runtime.GOOS, runtime.GOARCH }
func (OSSources) NumCPU() int                { return runtime.NumCPU() }
func (OSSources) ReadFile(p string) ([]byte, error) {
	return os.ReadFile(p)
}
func (OSSources) Glob(p string) ([]string, error) { return filepath.Glob(p) }

func (OSSources) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	if name == "nvidia-smi" && runtime.GOOS == "linux" {
		// A read whose ctx is smiloop.Sampled, while a user holds the loop, is answered from the
		// long-lived child's latest block; every other read (and any fallback) is the one-shot.
		if b, ok := smiloop.ReadFor(ctx, args); ok {
			return b, nil
		}
	}
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	path, err := exec.LookPath(name)
	if err != nil {
		for _, alt := range fallbackPaths(name) {
			if _, serr := os.Stat(alt); serr == nil {
				path, err = alt, nil
				break
			}
		}
		if err != nil {
			return nil, err
		}
	}
	cmd := exec.CommandContext(ctx, path, args...)
	hideWindow(cmd)
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return out.Bytes(), err
	}
	return out.Bytes(), nil
}

func fallbackPaths(name string) []string {
	if runtime.GOOS == "windows" && name == "nvidia-smi" {
		root := os.Getenv("SystemRoot")
		if root == "" {
			root = `C:\Windows`
		}
		return []string{
			filepath.Join(root, "System32", "nvidia-smi.exe"),
			`C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe`,
		}
	}
	if runtime.GOOS == "windows" && name == "powershell" {
		root := os.Getenv("SystemRoot")
		if root == "" {
			root = `C:\Windows`
		}
		return []string{filepath.Join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")}
	}
	return nil
}
