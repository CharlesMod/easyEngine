package honest

import "time"

func Same() {
	time.Sleep(1)        // watchdog for silence
	_ = time.After(1)    // network timeout
	_ = time.NewTimer(1) // named moment
}

func Above() {
	// watchdog for silence
	time.Sleep(1)
	// network timeout
	_ = time.NewTicker(1)
	// named moment
	//
	_ = time.Tick(1)
	// network timeout
	time.AfterFunc(1, func() {})
}
