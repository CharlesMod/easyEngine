package smiloop

import (
	"bytes"
	"strconv"
	"strings"
)

// A block parser is fed the child's lines one at a time and says when a block is whole. The three
// queries the engine loops each print a block of a different shape, and none of the shapes needs a
// timer to know its end:
//
//   - csv: `--query-gpu=count,<fields> --format=csv,noheader,nounits`. One line per GPU, and the
//     first field, `count`, is the number of GPUs, so a block is whole when that many lines are in.
//     The count is stripped from the block handed out, which is what the caller's own query printed.
//   - xml: `-q -x`. A block ends at the closing tag `</nvidia_smi_log>`.
//   - pids: `-q -d PIDS`. A block starts at the `==============NVSMI LOG` banner, says `Attached
//     GPUs : N`, then one `GPU <busid>` section per GPU. The section's lines run without a blank
//     line in them (checked on a one-shot's output), and a blank line closes the block's last
//     section, so the block is whole at the first blank line after the N-th `GPU` line has had a
//     line of its own after it.
type parser interface {
	feed(line string) (block []byte, whole bool)
}

type csvParser struct {
	want  int
	lines []string
}

func (p *csvParser) feed(line string) ([]byte, bool) {
	line = strings.TrimSpace(line)
	if line == "" {
		return nil, false
	}
	n, rest, ok := strings.Cut(line, ",")
	c, err := strconv.Atoi(strings.TrimSpace(n))
	if !ok || err != nil || c < 1 {
		p.want, p.lines = 0, nil // not a line of ours (a message, or a one-shot's lines): start over
		return nil, false
	}
	if p.want != c {
		p.want, p.lines = c, nil
	}
	p.lines = append(p.lines, strings.TrimSpace(rest))
	if len(p.lines) < p.want {
		return nil, false
	}
	b := []byte(strings.Join(p.lines, "\n") + "\n")
	p.want, p.lines = 0, nil
	return b, true
}

type xmlParser struct{ buf bytes.Buffer }

func (p *xmlParser) feed(line string) ([]byte, bool) {
	if strings.HasPrefix(line, "<?xml") {
		p.buf.Reset()
	}
	p.buf.WriteString(line)
	p.buf.WriteByte('\n')
	if strings.HasPrefix(strings.TrimSpace(line), "</nvidia_smi_log>") {
		b := append([]byte(nil), p.buf.Bytes()...)
		p.buf.Reset()
		return b, true
	}
	return nil, false
}

type pidsParser struct {
	buf     bytes.Buffer
	in      bool
	want    int  // Attached GPUs
	gpus    int  // "GPU <busid>" lines seen
	content bool // a line after the last GPU line
}

func (p *pidsParser) feed(line string) ([]byte, bool) {
	t := strings.TrimRight(line, "\r")
	if strings.HasPrefix(t, "==============NVSMI LOG") {
		p.buf.Reset()
		p.in, p.want, p.gpus, p.content = true, 0, 0, false
	}
	if !p.in {
		return nil, false
	}
	p.buf.WriteString(t)
	p.buf.WriteByte('\n')
	switch {
	case strings.HasPrefix(t, "Attached GPUs"):
		if _, v, ok := strings.Cut(t, ":"); ok {
			p.want, _ = strconv.Atoi(strings.TrimSpace(v))
		}
	case strings.HasPrefix(t, "GPU ") && !strings.Contains(t, " : "):
		p.gpus, p.content = p.gpus+1, false
	case strings.TrimSpace(t) == "":
		if p.want > 0 && p.gpus >= p.want && p.content {
			b := append([]byte(nil), p.buf.Bytes()...)
			p.in = false
			return b, true
		}
	default:
		if p.gpus > 0 {
			p.content = true
		}
	}
	return nil, false
}
