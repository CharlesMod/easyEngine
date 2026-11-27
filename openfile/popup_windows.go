package openfile

import "golang.org/x/sys/windows/registry"

// windowsDefaultBrowserCommand is the shell\open\command of the program registered for https:
// the user's choice (UserChoice\ProgId), then that class's command line. Best effort: "" when
// the registry says nothing.
func windowsDefaultBrowserCommand() string {
	k, err := registry.OpenKey(registry.CURRENT_USER,
		`Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice`, registry.QUERY_VALUE)
	if err != nil {
		return ""
	}
	progID, _, err := k.GetStringValue("ProgId")
	k.Close()
	if err != nil || progID == "" {
		return ""
	}
	c, err := registry.OpenKey(registry.CLASSES_ROOT, progID+`\shell\open\command`, registry.QUERY_VALUE)
	if err != nil {
		return ""
	}
	defer c.Close()
	cmd, _, err := c.GetStringValue("")
	if err != nil {
		return ""
	}
	return cmd
}
