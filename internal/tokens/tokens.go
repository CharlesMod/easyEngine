// Package tokens keeps the bearer tokens a non-loopback engine accepts.
//
// They live in <data dir>/tokens.json (mode 0600): the master token in clear, because
// `easy-engine token` prints it, and a list of further tokens as {id, name, created, sha256}
// where sha256 is the hash of the token (the token itself is shown once, when it is issued).
// The HTTP gate calls only Check; it never sees the file layout. There are no expiry fields
// and no timers.
package tokens

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/CharlesMod/easyEngine/lifecycle"
)

// FileName is the token file under the data dir.
const FileName = "tokens.json"

// MasterID is the client id Check reports for the master token.
const MasterID = "master"

// Entry is one issued token. The token itself is not kept, only its SHA-256 (hex).
type Entry struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Created string `json:"created"`
	SHA256  string `json:"sha256"`
}

type fileData struct {
	Master string  `json:"master"`
	Tokens []Entry `json:"tokens"`
}

// Store is the token file of one data dir. Methods are safe for concurrent use; each one reads
// the file afresh, so a `token --new` run by another process takes effect at once.
type Store struct {
	path string
	mu   sync.Mutex
}

// Open returns the store for a data dir. It touches nothing on disk.
func Open(dir string) *Store { return &Store{path: filepath.Join(dir, FileName)} }

// Path is the token file's path.
func (s *Store) Path() string { return s.path }

func newToken() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

func hash(tok string) [32]byte { return sha256.Sum256([]byte(tok)) }

// load reads the file; a missing file is an empty one (exists false).
func (s *Store) load() (d fileData, exists bool, err error) {
	b, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return fileData{}, false, nil
	}
	if err != nil {
		return fileData{}, false, err
	}
	if err := json.Unmarshal(b, &d); err != nil {
		return fileData{}, true, fmt.Errorf("%s: %w", s.path, err)
	}
	return d, true, nil
}

func (s *Store) save(d fileData) error {
	b, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return err
	}
	return lifecycle.WriteFileAtomic(s.path, b, 0o600)
}

// Master returns the master token, making it at first need. Two processes asking at once get
// the same token: the new file is hard-linked into place, which fails if one is already there.
func (s *Store) Master() (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	d, exists, err := s.load()
	if err != nil {
		return "", err
	}
	if d.Master != "" {
		return d.Master, nil
	}
	tok, err := newToken()
	if err != nil {
		return "", err
	}
	d.Master = tok
	if exists { // a file with no master (only issued tokens): add one
		return tok, s.save(d)
	}
	if err := s.createExclusive(d); err != nil {
		if errors.Is(err, os.ErrExist) { // another process made it first
			d2, _, lerr := s.load()
			if lerr != nil {
				return "", lerr
			}
			if d2.Master != "" {
				return d2.Master, nil
			}
		}
		return "", err
	}
	return tok, nil
}

// createExclusive writes d as a new file, failing with os.ErrExist if the file is there.
func (s *Store) createExclusive(d fileData) error {
	b, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, FileName+".tmp-*") // mode 0600
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	_, werr := f.Write(b)
	if werr == nil {
		werr = f.Sync()
	}
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		return werr
	}
	if err := os.Link(tmp, s.path); err != nil {
		if errors.Is(err, os.ErrExist) {
			return err
		}
		// No hard links on this filesystem: rename, if nobody got there first.
		if _, serr := os.Stat(s.path); serr == nil {
			return os.ErrExist
		}
		return os.Rename(tmp, s.path)
	}
	return nil
}

// NewMaster replaces the master token with a new one and returns it. Issued tokens stay.
func (s *Store) NewMaster() (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	d, _, err := s.load()
	if err != nil {
		return "", err
	}
	tok, err := newToken()
	if err != nil {
		return "", err
	}
	d.Master = tok
	return tok, s.save(d)
}

// Check reports whether bearer is a token this store accepts, and which client it belongs to:
// MasterID for the master token, the entry's id for an issued one. Hashes are compared in
// constant time. A missing or unreadable file accepts nothing.
func (s *Store) Check(bearer string) (clientID string, ok bool) {
	if bearer == "" {
		return "", false
	}
	s.mu.Lock()
	d, _, err := s.load()
	s.mu.Unlock()
	if err != nil {
		return "", false
	}
	h := hash(bearer)
	clientID, ok = "", false
	if d.Master != "" {
		mh := hash(d.Master)
		if subtle.ConstantTimeCompare(h[:], mh[:]) == 1 {
			clientID, ok = MasterID, true
		}
	}
	for _, e := range d.Tokens { // no early exit: every entry is compared
		want, err := hex.DecodeString(e.SHA256)
		if err != nil || len(want) != len(h) {
			continue
		}
		if subtle.ConstantTimeCompare(h[:], want) == 1 && !ok {
			clientID, ok = e.ID, true
		}
	}
	return clientID, ok
}

// Issue makes a new token for a named client and returns its id and the token, which is not
// kept and cannot be shown again. Nothing calls it yet.
func (s *Store) Issue(name string) (id, token string, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	d, _, err := s.load()
	if err != nil {
		return "", "", err
	}
	token, err = newToken()
	if err != nil {
		return "", "", err
	}
	idb := make([]byte, 6)
	if _, err := rand.Read(idb); err != nil {
		return "", "", err
	}
	id = hex.EncodeToString(idb)
	h := hash(token)
	d.Tokens = append(d.Tokens, Entry{ID: id, Name: name, Created: time.Now().UTC().Format(time.RFC3339), SHA256: hex.EncodeToString(h[:])})
	if err := s.save(d); err != nil {
		return "", "", err
	}
	return id, token, nil
}

// Revoke removes an issued token by id. The master token is replaced with NewMaster, not revoked.
func (s *Store) Revoke(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	d, _, err := s.load()
	if err != nil {
		return err
	}
	for i, e := range d.Tokens {
		if e.ID == id {
			d.Tokens = append(d.Tokens[:i], d.Tokens[i+1:]...)
			return s.save(d)
		}
	}
	return fmt.Errorf("no token with id %q", id)
}

// List returns the issued tokens (never the master, never a token value).
func (s *Store) List() ([]Entry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	d, _, err := s.load()
	return d.Tokens, err
}

// Existing returns the master token if the file has one; it never creates anything. The
// engine's own clients use it to find the token of a running engine.
func (s *Store) Existing() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	d, _, err := s.load()
	return d.Master, err == nil && d.Master != ""
}
