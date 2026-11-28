package api_test

import (
	"fmt"
	"net"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// lookRemoteAPI is a tokened listener on the look engine whose API is listened to before it serves:
// each count of remote bodies being read, and each count of remote bytes kept.
func lookRemoteAPI(t *testing.T) (*testkit.Env, *run.Supervisor, *httptest.Server, chan int32, *keptLog) {
	t.Helper()
	e, s, _ := lookEnvE(t, func(e *testkit.Env) {
		e.Prof.CtxPerSlot.Max, e.Prof.CtxPerSlot.Floor = 8192, 4096
	}, nil, true)
	a := &api.API{S: s, AnyHost: true, Tokens: newRevocable()}
	reads := make(chan int32, 64)
	a.SayReads(func(n int32) { reads <- n })
	kl := &keptLog{}
	a.SayKept(kl.say)
	rs := httptest.NewServer(a.Handler())
	t.Cleanup(rs.Close)
	return e, s, rs, reads, kl
}

type keptLog struct {
	mu   sync.Mutex
	seen []int64
}

func (k *keptLog) say(n int64) { k.mu.Lock(); k.seen = append(k.seen, n); k.mu.Unlock() }
func (k *keptLog) reset()      { k.mu.Lock(); k.seen = nil; k.mu.Unlock() }
func (k *keptLog) got() []int64 {
	k.mu.Lock()
	defer k.mu.Unlock()
	return append([]int64(nil), k.seen...)
}

// A remote look's bytes are counted without a gap and without a double: its body is counted when
// read, the request built from it is counted before the body's count is given back, the proxy
// counts nothing more, and both are given back when the look ends. The counts said, in order, are
// exactly body, body+built, built, 0. Without the hand-over's count the third is 0; without the
// body's give the last is not 0; with the proxy reading the built request again a count of twice
// built (or more) appears.
func TestLookRemoteBytesCountedOnce(t *testing.T) {
	e, s, rs, _, kl := lookRemoteAPI(t)
	growRemote(t, e, s, rs, 1, "a", 1)
	body := oneLook(pngB64(t, 300, 200), "")
	kl.reset()
	if code, b := postLook(t, rs.URL, body, as(1)); code != 200 {
		t.Fatalf("look: %d %s", code, b)
	}
	seen := kl.got()
	if len(seen) != 4 {
		t.Fatalf("counts said %v, want body, body+built, built, 0", seen)
	}
	n := int64(len(body))
	built := seen[2]
	if seen[0] != n || built <= 0 || seen[1] != n+built || seen[3] != 0 {
		t.Fatalf("counts said %v (body %d), want %d, %d+built, built, 0", seen, n, n, n)
	}
}

// The look's gate is taken after its body is read: two senders stalled mid-body (more than the
// looks decoded at once) hold only their reads, and another caller's look is answered meanwhile.
func TestLookSlowSendersHoldNoGate(t *testing.T) {
	e, s, rs, reads, _ := lookRemoteAPI(t)
	growRemote(t, e, s, rs, 1, "a", 1)
	for i := 2; i <= 3; i++ {
		conn, err := net.Dial("tcp", rs.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { conn.Close() })
		fmt.Fprintf(conn, "POST /engine/v1/look HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer tok-%d\r\nX-Easy-Engine: 1\r\nContent-Type: application/json\r\nContent-Length: 1000\r\n\r\n{", i)
	}
	// the stalled reads say nothing to any hub: the test waits on the API's own word of the count
	for n := int32(0); n < 2; {
		select {
		case n = <-reads:
		case <-time.After(10 * time.Second): // network timeout: the two reads begin long before this
			t.Fatal("timed out waiting for two bodies being read")
		}
	}
	if code, b := postLook(t, rs.URL, oneLook(pngB64(t, 40, 30), ""), as(1)); code != 200 {
		t.Fatalf("a look beside two stalled senders: %d %s", code, b)
	}
}
