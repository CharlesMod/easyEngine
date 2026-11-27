// easy-engine: the one-click local LLM runtime (see docs/ENGINE.md).
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
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/dirs"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/tray"
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
  easy-engine [app]   a tray icon, stays open and waits for a connection (settings in <data dir>/settings.json;
                        the serve flags below override it for one run)
  easy-engine serve   --profile F --parent-pid P [--port N] [--linger D]
                        [--yield-to-games [--yield-except IDS] [--yield-ignore NAMES] [--yield-floor-mib N]
                         [--yield-min-mib N] [--yield-grace D] [--yield-steady D] [--yield-strict]]
  easy-engine setup   [--profile F]
  easy-engine token   [--new]
  easy-engine probe
  easy-engine plan    [--profile F]
  easy-engine gpu     [--profile F] [--host-pid N] [--json]
  easy-engine install --model ID [--yes]
  easy-engine import  PATH
  easy-engine status | stop
  easy-engine version
`)
}

// main hands the app to tray.Main: macOS runs its UI loop on the main thread and the rest of
// the program in a goroutine; Linux and Windows just call it. Every other command (serve, the
// CLI) runs plainly: it shows no icon, so it starts no UI loop (a shell over ssh has none).
func main() {
	if len(os.Args) >= 2 && os.Args[1] != "app" {
		os.Exit(realMain())
	}
	var code int
	tray.Main(func() { code = realMain() })
	os.Exit(code)
}

func realMain() int {
	cmd, args := "app", os.Args[1:] // no command: the app
	implicit := len(os.Args) < 2
	if !implicit {
		cmd, args = os.Args[1], os.Args[2:]
	}
	switch cmd {
	case "app", "serve", "setup", "install", "import", "token":
		// The engine deletes inside its data dir, so a dir it does not own is refused up front.
		if err := dirs.Check(); err != nil {
			fmt.Fprintln(os.Stderr, "easy-engine:", err)
			return 2
		}
	}
	var code int
	switch cmd {
	case "app":
		code = cmdApp(args, implicit)
	case "serve":
		code = cmdServe(args, false)
	case "setup":
		code = cmdServe(args, true)
	case "probe":
		code = cmdProbe()
	case "plan":
		code = cmdPlan(args)
	case "gpu":
		code = cmdGPU(args)
	case "install":
		code = cmdInstall(args)
	case "import":
		code = cmdImport(args)
	case "token":
		code = cmdToken(args)
	case "status":
		code = cmdStatus()
	case "stop":
		code = cmdStop()
	case "version", "--version", "-v":
		fmt.Println("easy-engine", version)
	default:
		usage()
		code = 2
	}
	return code
}

func newFlags(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	return fs
}

// splitList splits a comma-separated flag value, dropping empty items.
func splitList(s string) []string {
	var out []string
	for _, f := range strings.Split(s, ",") {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	return out
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
	if tg, ok := app.Find(dirs.Data()); ok {
		return installVia(tg, *model, *backend)
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

func installVia(tg app.Target, model, backend string) int {
	c := &http.Client{Timeout: 10 * time.Second}
	body, _ := json.Marshal(map[string]string{"model": model, "backend": backend})
	req, err := tg.NewRequest("POST", "/engine/v1/install", bytes.NewReader(body))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
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
	// get reads the engine's status; a transport error, a non-200 answer or an unreadable body
	// is an error (a dead engine must not look like a finished install).
	get := func() (run.Status, error) {
		var st run.Status
		req, err := tg.NewRequest("GET", "/engine/v1/status", nil)
		if err != nil {
			return st, err
		}
		r, err := c.Do(req)
		if err != nil {
			return st, err
		}
		defer r.Body.Close()
		if r.StatusCode != 200 {
			return st, fmt.Errorf("status: %s", r.Status)
		}
		if err := json.NewDecoder(r.Body).Decode(&st); err != nil {
			return st, fmt.Errorf("status: %v", err)
		}
		return st, nil
	}
	done := make(chan error, 1)
	go func() {
		time.Sleep(500 * time.Millisecond)
		for {
			st, err := get()
			if err != nil {
				done <- fmt.Errorf("lost the engine: %v", err)
				return
			}
			if st.Phase != "downloading" && st.Phase != "installing" {
				switch st.Phase {
				case "error":
					done <- fmt.Errorf("%s", st.Message)
				case "stopping":
					done <- fmt.Errorf("the engine is stopping")
				default:
					done <- nil
				}
				return
			}
			time.Sleep(500 * time.Millisecond)
		}
	}()
	return waitBar(done, func() run.Status { st, _ := get(); return st })
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
		fmt.Fprintln(os.Stderr, "usage: easy-engine import PATH")
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

func engineTarget() (app.Target, bool) { return app.Find(dirs.Data()) }

// cmdToken prints the engine's bearer token (made if absent); --new replaces it.
func cmdToken(args []string) int {
	fs := newFlags("token")
	fresh := fs.Bool("new", false, "replace the token with a new one (clients must be given the new one)")
	if fs.Parse(args) != nil {
		return 2
	}
	st := tokens.Open(dirs.Data())
	var tok string
	var err error
	if *fresh {
		tok, err = st.NewMaster()
	} else {
		tok, err = st.Master()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "token:", err)
		return 1
	}
	fmt.Println(tok)
	return 0
}

func cmdStatus() int {
	tg, ok := engineTarget()
	if !ok {
		fmt.Println(`{"running": false}`)
		return 1
	}
	req, _ := tg.NewRequest("GET", "/engine/v1/status", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer resp.Body.Close()
	io.Copy(os.Stdout, resp.Body)
	return 0
}

func cmdStop() int {
	d, err := lifecycle.ReadLock(filepath.Join(dirs.Data(), "engine.lock"))
	tg, ok := engineTarget()
	if err != nil || !ok {
		fmt.Println("No engine is running.")
		return 0
	}
	req, _ := tg.NewRequest("POST", "/engine/v1/shutdown", nil)
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

// cmdGPU prints the adapters, the processes holding memory on them and the reserve the engine
// plans around. With an engine running it asks that engine (which knows its own server's pid);
// otherwise it senses here.
func cmdGPU(args []string) int {
	fs := newFlags("gpu")
	profile := fs.String("profile", "", "game profile file")
	hostPID := fs.Int("host-pid", 0, "the host app's pid (default: the profile's)")
	asJSON := fs.Bool("json", false, "print JSON")
	if fs.Parse(args) != nil {
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	var rep gpusense.GPUReport
	if tg, ok := engineTarget(); ok {
		if *hostPID > 0 {
			body := strings.NewReader(fmt.Sprintf(`{"pid": %d}`, *hostPID))
			req, _ := tg.NewRequest("POST", "/engine/v1/host", body)
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				return 1
			}
			resp.Body.Close()
			if resp.StatusCode != http.StatusNoContent {
				fmt.Fprintln(os.Stderr, "host:", resp.Status)
				return 1
			}
		}
		req, _ := tg.NewRequest("GET", "/engine/v1/gpu?refresh=1", nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 || json.NewDecoder(resp.Body).Decode(&rep) != nil {
			fmt.Fprintln(os.Stderr, "gpu: the engine did not answer with a report:", resp.Status)
			return 1
		}
	} else {
		s, err := supervisor(*profile, true)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		defer s.Shutdown()
		if *hostPID > 0 {
			if err := s.SetHost(run.HostSet{PID: *hostPID}); err != nil {
				fmt.Fprintln(os.Stderr, err)
				return 1
			}
		}
		if rep, err = s.GPUReport(ctx, true); err != nil {
			fmt.Fprintln(os.Stderr, "gpu:", err)
			return 1
		}
	}
	if *asJSON {
		return printJSON(rep)
	}
	fmt.Print(formatGPU(rep))
	return 0
}

func gb(mib int) string { return fmt.Sprintf("%.1f GB", float64(mib)/1024) }

func formatGPU(r gpusense.GPUReport) string {
	var b strings.Builder
	if len(r.Adapters) == 0 {
		b.WriteString("No graphics adapter found.\n")
	}
	for _, a := range r.Adapters {
		kind := "discrete"
		switch {
		case a.Unified:
			kind = "unified memory"
		case a.Integrated:
			kind = "integrated"
		}
		role := ""
		if a.Index == r.EngineAdapter {
			role += " [engine]"
		}
		if a.Index == r.HostAdapter {
			role += " [host]"
		}
		used := "used unknown"
		if a.UsedMiB >= 0 {
			used = fmt.Sprintf("%d MiB used", a.UsedMiB)
		}
		fmt.Fprintf(&b, "adapter %d: %s (%s, %s) %d MiB dedicated, %d MiB shared, %s%s\n", a.Index, a.Name, a.Vendor, kind, a.DedicatedMiB, a.SharedMiB, used, role)
	}
	if len(r.Processes) > 0 {
		b.WriteString("processes:\n")
	}
	for _, p := range r.Processes {
		tag := ""
		if p.Ours {
			tag = " (engine)"
		} else if p.Host {
			tag = " (host)"
		}
		name := p.Name
		if name == "" {
			name = "?"
		}
		fmt.Fprintf(&b, "  pid %d %s on adapter %d: %d MiB dedicated, %d MiB shared%s\n", p.PID, name, p.Adapter, p.DedicatedMiB, p.SharedMiB, tag)
	}
	if r.EngineAdapter < 0 {
		b.WriteString("engine: CPU (no usable adapter)\n")
	} else {
		x := r.Reserve
		fmt.Fprintf(&b, "reserve: others %d MiB (%s), host %d MiB (%s), margin %d MiB, of %d MiB\n", x.OthersMiB, x.OthersBasis, x.HostMiB, x.HostBasis, x.MarginMiB, x.TotalMiB)
		fmt.Fprintf(&b, "free for the model: %s (%d MiB)\n", gb(r.FreeForModelMiB), r.FreeForModelMiB)
	}
	if r.OthersNote != "" {
		b.WriteString(r.OthersNote + "\n")
	}
	var keys []string
	for k := range r.Sources {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if r.Sources[k] != "ok" {
			fmt.Fprintf(&b, "source %s: %s\n", k, r.Sources[k])
		}
	}
	return b.String()
}
