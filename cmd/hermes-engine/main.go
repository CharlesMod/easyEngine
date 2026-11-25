// hermes-engine: the one-click local LLM runtime (see docs/ENGINE.md).
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"hermesengine/app"
	"hermesengine/dirs"
	"hermesengine/lifecycle"
	"hermesengine/probe"
	"hermesengine/run"
)

// version is set with -ldflags "-X main.version=...".
var version = "dev"

type pidList []int

func (p *pidList) String() string { return fmt.Sprint(*p) }
func (p *pidList) Set(v string) error {
	var n int
	if _, err := fmt.Sscan(v, &n); err != nil || n <= 0 {
		return fmt.Errorf("bad pid %q", v)
	}
	*p = append(*p, n)
	return nil
}

func usage() {
	fmt.Fprint(os.Stderr, `usage:
  hermes-engine serve   --profile F --parent-pid P [--port N] [--linger D]
  hermes-engine setup   [--profile F]
  hermes-engine probe
  hermes-engine plan    [--profile F]
  hermes-engine install --model ID [--yes]
  hermes-engine import  PATH
  hermes-engine status | stop
  hermes-engine version
`)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd, args := os.Args[1], os.Args[2:]
	var code int
	switch cmd {
	case "serve":
		code = cmdServe(args, false)
	case "setup":
		code = cmdServe(args, true)
	case "probe":
		code = cmdProbe()
	case "plan":
		code = cmdPlan(args)
	case "install":
		code = cmdInstall(args)
	case "import":
		code = cmdImport(args)
	case "status":
		code = cmdStatus()
	case "stop":
		code = cmdStop()
	case "version", "--version", "-v":
		fmt.Println("hermes-engine", version)
	default:
		usage()
		code = 2
	}
	os.Exit(code)
}

func newFlags(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	return fs
}

func cmdServe(args []string, setup bool) int {
	fs := newFlags("serve")
	profile := fs.String("profile", "", "game profile file")
	port := fs.Int("port", 0, "control API port (default 47480)")
	host := fs.String("host", "127.0.0.1", "listen address")
	linger := fs.Duration("linger", 0, "keep running this long after the last parent exits")
	var parents pidList
	fs.Var(&parents, "parent-pid", "game pid to watch (repeatable)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	lingerSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "linger" {
			lingerSet = true
		}
	})
	if !setup && len(parents) == 0 {
		fmt.Fprintln(os.Stderr, "serve needs --parent-pid (use `setup` for a run without a game)")
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	o := app.Options{
		ProfilePath: *profile, ParentPIDs: parents, Host: *host, Port: *port,
		Linger: *linger, LingerSet: lingerSet, Version: version,
		DetachStdio: !setup,
		Started: func(p int) {
			if setup {
				fmt.Printf("Setup page: http://127.0.0.1:%d/\n", p)
				openBrowser(fmt.Sprintf("http://127.0.0.1:%d/", p))
			}
		},
	}
	res, err := app.Serve(ctx, o)
	if err != nil {
		fmt.Fprintln(os.Stderr, "hermes-engine:", err)
		return 1
	}
	if res.Attached {
		if setup {
			u := fmt.Sprintf("http://127.0.0.1:%d/", res.Port)
			fmt.Println("Engine already running:", u)
			openBrowser(u)
		}
		return 0
	}
	return 0
}

func openBrowser(url string) {
	var c *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		c = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		c = exec.Command("open", url)
	default:
		c = exec.Command("xdg-open", url)
	}
	if err := c.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "could not open a browser:", err, "- open", url, "yourself")
		return
	}
	go c.Wait()
}

func cmdProbe() int {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	hw, err := probe.Probe(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "probe:", err)
		return 1
	}
	return printJSON(hw)
}

func printJSON(v any) int {
	b, _ := json.MarshalIndent(v, "", "  ")
	fmt.Println(string(b))
	return 0
}

func supervisor(profile string, noAuto bool) (*run.Supervisor, error) {
	p, err := run.LoadProfile(profile)
	if err != nil {
		return nil, err
	}
	return run.New(run.Config{DataDir: dirs.Data(), Profile: p, Version: version, NoAutoStart: noAuto})
}

func cmdPlan(args []string) int {
	fs := newFlags("plan")
	profile := fs.String("profile", "", "game profile file")
	if fs.Parse(args) != nil {
		return 2
	}
	s, err := supervisor(*profile, true)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer s.Shutdown()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	p, err := s.Plan(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "plan:", err)
		return 1
	}
	return printJSON(p)
}

func cmdInstall(args []string) int {
	fs := newFlags("install")
	model := fs.String("model", "", "model id from the plan")
	profile := fs.String("profile", "", "game profile file")
	backend := fs.String("backend", "", "backend id (default: automatic)")
	yes := fs.Bool("yes", false, "do not ask")
	if fs.Parse(args) != nil || *model == "" {
		fmt.Fprintln(os.Stderr, "install needs --model ID")
		return 2
	}
	if port, ok := app.Running(dirs.Data()); ok {
		return installVia(port, *model, *backend)
	}
	s, err := supervisor(*profile, true)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer s.Shutdown()
	if !*yes {
		fmt.Printf("Install %s into %s? [y/N] ", *model, dirs.Data())
		var a string
		fmt.Scanln(&a)
		if !strings.HasPrefix(strings.ToLower(a), "y") {
			return 1
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	done := make(chan error, 1)
	go func() { done <- s.InstallSync(ctx, *model, *backend) }()
	return waitBar(done, func() run.Status { return s.Status() })
}

func installVia(port int, model, backend string) int {
	c := &http.Client{Timeout: 10 * time.Second}
	base := fmt.Sprintf("http://127.0.0.1:%d/engine/v1/", port)
	body, _ := json.Marshal(map[string]string{"model": model, "backend": backend})
	req, _ := http.NewRequest("POST", base+"install", bytes.NewReader(body))
	req.Header.Set("X-Hermes-Engine", "1")
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.Do(req)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 202 {
		fmt.Fprintf(os.Stderr, "install: %s %s\n", resp.Status, b)
		return 1
	}
	get := func() run.Status {
		var st run.Status
		if r, err := c.Get(base + "status"); err == nil {
			json.NewDecoder(r.Body).Decode(&st)
			r.Body.Close()
		}
		return st
	}
	done := make(chan error, 1)
	go func() {
		time.Sleep(500 * time.Millisecond)
		for {
			st := get()
			if st.Phase != "downloading" && st.Phase != "installing" {
				if st.Phase == "error" {
					done <- fmt.Errorf("%s", st.Message)
				} else {
					done <- nil
				}
				return
			}
			time.Sleep(500 * time.Millisecond)
		}
	}()
	return waitBar(done, get)
}

func waitBar(done chan error, status func() run.Status) int {
	t := time.NewTicker(300 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case err := <-done:
			fmt.Fprint(os.Stderr, "\r\033[K")
			if err != nil {
				fmt.Fprintln(os.Stderr, "install failed:", err)
				return 1
			}
			fmt.Println("Installed.")
			return 0
		case <-t.C:
			st := status()
			fmt.Fprint(os.Stderr, "\r\033[K"+renderBar(st))
		}
	}
}

func renderBar(st run.Status) string {
	p := st.Progress
	if p == nil || p.TotalBytes <= 0 {
		return st.Message
	}
	const w = 30
	frac := float64(p.DoneBytes) / float64(p.TotalBytes)
	if frac > 1 {
		frac = 1
	}
	n := int(frac * w)
	return fmt.Sprintf("%-22s [%s%s] %3.0f%% %s/%s %s/s ETA %ds", trunc(p.Step, 22), strings.Repeat("#", n), strings.Repeat(".", w-n),
		frac*100, human(p.DoneBytes), human(p.TotalBytes), human(int64(p.RateBps)), p.ETAS)
}

func trunc(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func human(b int64) string {
	f := float64(b)
	for _, u := range []string{"B", "KB", "MB", "GB"} {
		if f < 1024 {
			return fmt.Sprintf("%.1f%s", f, u)
		}
		f /= 1024
	}
	return fmt.Sprintf("%.1fTB", f)
}

func cmdImport(args []string) int {
	if len(args) != 1 {
		fmt.Fprintln(os.Stderr, "usage: hermes-engine import PATH")
		return 2
	}
	s, err := supervisor("", true)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer s.Shutdown()
	if err := s.ImportSync(context.Background(), args[0]); err != nil {
		fmt.Fprintln(os.Stderr, "import:", err)
		return 1
	}
	fmt.Println("Imported.")
	return 0
}

func enginePort() (int, bool) { return app.Running(dirs.Data()) }

func cmdStatus() int {
	port, ok := enginePort()
	if !ok {
		fmt.Println(`{"running": false}`)
		return 1
	}
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/engine/v1/status", port))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer resp.Body.Close()
	io.Copy(os.Stdout, resp.Body)
	return 0
}

func cmdStop() int {
	d, err := lifecycle.ReadLock(dirs.Data() + string(os.PathSeparator) + "engine.lock")
	port, ok := enginePort()
	if err != nil || !ok {
		fmt.Println("No engine is running.")
		return 0
	}
	req, _ := http.NewRequest("POST", fmt.Sprintf("http://127.0.0.1:%d/engine/v1/shutdown", port), nil)
	req.Header.Set("X-Hermes-Engine", "1")
	if resp, err := http.DefaultClient.Do(req); err == nil {
		resp.Body.Close()
	}
	for i := 0; i < 100; i++ {
		if !lifecycle.PidAlive(d.Pid) {
			fmt.Println("Stopped.")
			return 0
		}
		time.Sleep(100 * time.Millisecond)
	}
	fmt.Fprintln(os.Stderr, "engine did not stop in 10 s")
	return 1
}
