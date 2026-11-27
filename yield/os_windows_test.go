package yield

import (
	"errors"
	"reflect"
	"testing"
)

type fakeReg struct {
	id  uint32
	err error
}

func (f fakeReg) RunningAppID() (uint32, error) { return f.id, f.err }

func TestRegistryLaunches(t *testing.T) {
	tests := []struct {
		name string
		reg  fakeReg
		want []Launch
	}{
		{"zero", fakeReg{}, nil},
		{"error", fakeReg{id: 5, err: errors.New("no key")}, nil},
		{"running", fakeReg{id: 1234}, []Launch{{Kind: "steam-registry", AppID: "1234"}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := registryLaunches(tc.reg); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v want %v", got, tc.want)
			}
			if got := newWindowsOS(tc.reg).Launches(); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("OS.Launches got %v want %v", got, tc.want)
			}
		})
	}
}
