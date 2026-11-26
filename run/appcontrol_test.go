package run

import (
	"errors"
	"fmt"
	"os"
	"syscall"
	"testing"
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
