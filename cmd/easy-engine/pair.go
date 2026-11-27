package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/discover"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/pair"
)

// cmdFind lists the engines that answer one mDNS query on this network.
func cmdFind(args []string) int {
	fs := newFlags("find")
	wait := fs.Duration("wait", 2*time.Second, "how long to listen for answers")
	asJSON := fs.Bool("json", false, "print the engines as JSON")
	if fs.Parse(args) != nil {
		return 2
	}
	found, err := discover.Find(context.Background(), *wait)
	if err != nil {
		fmt.Fprintln(os.Stderr, "find:", err)
		return 1
	}
	return printFound(os.Stdout, found, *asJSON)
}

// findAddr is the address a client would call an engine on: an IPv4 address when it has one.
func findAddr(f discover.Found) string {
	host := strings.TrimSuffix(f.Host, ".")
	for _, ip := range f.Addrs {
		if ip.To4() != nil {
			host = ip.String()
			break
		}
	}
	if host == strings.TrimSuffix(f.Host, ".") && len(f.Addrs) > 0 {
		host = f.Addrs[0].String()
	}
	return net.JoinHostPort(host, strconv.Itoa(f.Port))
}

func printFound(w io.Writer, found []discover.Found, asJSON bool) int {
	if asJSON {
		type row struct {
			Name    string `json:"name"`
			ID      string `json:"engine_id"`
			API     int    `json:"api"`
			Address string `json:"address"`
			Host    string `json:"host"`
		}
		rows := []row{}
		for _, f := range found {
			rows = append(rows, row{f.Name, f.ID, f.API, findAddr(f), strings.TrimSuffix(f.Host, ".")})
		}
		b, _ := json.MarshalIndent(rows, "", "  ")
		fmt.Fprintln(w, string(b))
		return 0
	}
	if len(found) == 0 {
		fmt.Fprintln(w, "No engine answered on this network.")
		return 0
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tADDRESS\tENGINE ID\tAPI")
	for _, f := range found {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\n", f.Name, findAddr(f), f.ID, f.API)
	}
	tw.Flush()
	return 0
}

// cmdPair lists the pairing requests that wait, or answers one by its code.
func cmdPair(args []string) int {
	fs := newFlags("pair")
	approve := fs.String("approve", "", "allow the waiting request that shows this code")
	deny := fs.String("deny", "", "deny the waiting request that shows this code")
	if fs.Parse(args) != nil {
		return 2
	}
	tg, ok := engineTarget()
	if !ok {
		fmt.Fprintln(os.Stderr, "No engine is running.")
		return 1
	}
	return pairWith(tg, *approve, *deny, os.Stdout, os.Stderr)
}

func pairWith(tg app.Target, approve, deny string, out, errw io.Writer) int {
	if approve != "" && deny != "" {
		fmt.Fprintln(errw, "pair: give --approve or --deny, not both")
		return 2
	}
	if approve == "" && deny == "" {
		return pairList(tg, out, errw)
	}
	code, allow := approve, true
	if deny != "" {
		code, allow = deny, false
	}
	body, _ := json.Marshal(map[string]any{"code": strings.ReplaceAll(code, " ", ""), "allow": allow})
	var res struct {
		Name, Code, Error string
	}
	status, err := pairCall(tg, "POST", "/engine/v1/pairing/answer", body, &res)
	switch {
	case err != nil:
		fmt.Fprintln(errw, "pair:", err)
		return 1
	case status == http.StatusConflict:
		fmt.Fprintf(errw, "pair: two waiting requests show %s; answer on the setup page or in the tray\n", pair.SpacedCode(code))
		return 1
	case status == http.StatusNotFound:
		fmt.Fprintf(errw, "pair: no waiting request shows %s\n", pair.SpacedCode(code))
		return 1
	case status != http.StatusOK:
		fmt.Fprintf(errw, "pair: %d %s\n", status, res.Error)
		return 1
	}
	word := "Allowed"
	if !allow {
		word = "Denied"
	}
	fmt.Fprintf(out, "%s %s (%s).\n", word, res.Name, pair.SpacedCode(res.Code))
	return 0
}

func pairList(tg app.Target, out, errw io.Writer) int {
	var res struct {
		Offered  bool           `json:"offered"`
		Requests []pair.Request `json:"requests"`
		Error    string         `json:"error"`
	}
	status, err := pairCall(tg, "GET", "/engine/v1/pairing", nil, &res)
	if err != nil || status != http.StatusOK {
		fmt.Fprintln(errw, "pair:", err, status, res.Error)
		return 1
	}
	switch {
	case !res.Offered:
		fmt.Fprintln(out, "This engine offers no pairing (it listens on loopback, or with --open).")
		return 0
	case len(res.Requests) == 0:
		fmt.Fprintln(out, "No pairing request is waiting.")
		return 0
	}
	tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "CODE\tNAME\tFROM")
	for _, r := range res.Requests {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", pair.SpacedCode(r.Code), r.Name, r.From)
	}
	tw.Flush()
	fmt.Fprintln(out, "Compare the code with the one the client shows, then: easy-engine pair --approve <code> (or --deny <code>)")
	return 0
}

func pairCall(tg app.Target, method, path string, body []byte, v any) (int, error) {
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body)
	}
	req, err := tg.NewRequest(method, path, rd)
	if err != nil {
		return 0, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	_ = json.NewDecoder(resp.Body).Decode(v)
	return resp.StatusCode, nil
}

// tokenList prints the paired clients' tokens: id, name, created. Never a token or a hash.
func tokenList(st *tokens.Store, out io.Writer) int {
	list, err := st.List()
	if err != nil {
		fmt.Fprintln(os.Stderr, "token:", err)
		return 1
	}
	if len(list) == 0 {
		fmt.Fprintln(out, "No paired clients. (The master token is shown by: easy-engine token)")
		return 0
	}
	tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tNAME\tCREATED")
	for _, e := range list {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", e.ID, e.Name, e.Created)
	}
	tw.Flush()
	return 0
}

func tokenRevoke(st *tokens.Store, id string, out io.Writer) int {
	name := st.NameOf(id)
	if err := st.Revoke(id); err != nil {
		fmt.Fprintln(os.Stderr, "token:", err)
		return 1
	}
	fmt.Fprintf(out, "Revoked %s (%s). Its client must pair again.\n", id, name)
	return 0
}
