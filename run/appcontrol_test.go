package run

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestAppControlBlocked(t *testing.T) {
	var code uint32 = 0xC0E90002 // how Windows reports the exit status of a program whose DLL was refused
	if !blockedExitCode(int(code)) {
		t.Fatal("0xC0E90002 not recognised")
	}
	if blockedExitCode(1) || blockedExitCode(0) {
		t.Fatal("ordinary exit codes taken for a block")
	}
	start := &os.PathError{Op: "fork/exec", Path: "llama-server.exe", Err: syscall.Errno(4551)}
	if !appControlBlocked(fmt.Errorf("start: %w", start)) {
		t.Fatal("ERROR_SYSTEM_INTEGRITY_POLICY_VIOLATION at start not recognised")
	}
	if appControlBlocked(&os.PathError{Op: "open", Path: "x", Err: syscall.ENOENT}) || appControlBlocked(errors.New("x")) || appControlBlocked(nil) {
		t.Fatal("other errors taken for a block")
	}
}

func TestSmokeRunReportsTimeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a shell script")
	}
	bin := filepath.Join(t.TempDir(), "slow-server")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nsleep 5\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	_, err := smokeRun(context.Background(), 200*time.Millisecond, bin, "--list-devices")
	if err == nil || !strings.Contains(err.Error(), "timed out after 200ms") {
		t.Fatalf("err = %v, want a timeout named as one", err)
	}
	if smokeDevicesTimeout < 2*time.Minute {
		t.Fatalf("device listing limit %s is too short for a first CUDA load under an antivirus scan", smokeDevicesTimeout)
	}
}
