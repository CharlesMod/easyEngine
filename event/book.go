package event

import (
	"sort"
	"strings"
)

// Decl is one name's declaration in the book. The book is the API's schema:
// GET /engine/v1/book serves it.
type Decl struct {
	Name  string `json:"name"`
	Kind  Kind   `json:"kind"`
	Keyed bool   `json:"keyed,omitempty"` // the name has several, told apart by Key
	What  string `json:"what"`            // what it says and what its body is, in a line
}

// checkBook refuses a book with an empty, repeated or unkinded name.
func checkBook(book []Decl) (map[string]Decl, error) {
	m := make(map[string]Decl, len(book))
	for _, d := range book {
		switch {
		case d.Name == "":
			return nil, Refused{"a declaration", "it has no name"}
		case d.Kind != Fact && d.Kind != Reading && d.Kind != Happening:
			return nil, Refused{d.Name, "its kind \"" + string(d.Kind) + "\" is none of fact, reading, happening"}
		}
		if _, ok := m[d.Name]; ok {
			return nil, Refused{d.Name, "it is declared twice"}
		}
		m[d.Name] = d
	}
	return m, nil
}

// Book is every declaration, sorted by name.
func (h *Hub) Book() []Decl {
	out := make([]Decl, 0, len(h.book))
	for _, d := range h.book {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// declaredUnder says whether some declared name begins with prefix ("" is
// every name).
func (h *Hub) declaredUnder(prefix string) bool {
	for n := range h.book {
		if strings.HasPrefix(n, prefix) {
			return true
		}
	}
	return false
}
