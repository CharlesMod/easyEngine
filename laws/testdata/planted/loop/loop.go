package loop

import "time"

func Wait() {
	for i := 0; i < 3; i++ {
		time.Sleep(time.Second)
	}
}
