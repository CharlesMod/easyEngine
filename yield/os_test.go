package yield

import "testing"

func TestParseRegistryVDF(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"tabs and quotes", "\"Registry\"\n{\n\t\"HKCU\"\n\t{\n\t\t\"Software\"\n\t\t{\n\t\t\t\"Valve\"\n\t\t\t{\n\t\t\t\t\"Steam\"\n\t\t\t\t{\n\t\t\t\t\t\"RunningAppID\"\t\t\"1962700\"\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n}\n", "1962700"},
		{"zero", "\t\"RunningAppID\"\t\t\"0\"\n", ""},
		{"zeros", "\t\"RunningAppID\"\t\t\"000\"\n", ""},
		{"absent", "\"Registry\"\n{\n\t\"Other\"\t\"5\"\n}\n", ""},
		{"empty", "", ""},
		{"key case", "\t\"runningappid\"\t\t\"42\"\n", "42"},
		{"key mixed case", "\t\"RUNNINGAPPID\" \"43\"\r\n", "43"},
		{"non-numeric", "\t\"RunningAppID\"\t\t\"abc\"\n", ""},
		{"empty value", "\t\"RunningAppID\"\t\t\"\"\n", ""},
		{"first match wins", "\"RunningAppID\" \"0\"\n\"RunningAppID\" \"9\"\n", ""},
		{"other key with prefix", "\"LastRunningAppID\" \"9\"\n", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := ParseRegistryVDF([]byte(tc.in)); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}
