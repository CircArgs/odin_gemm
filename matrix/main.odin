package matmul
import "core:fmt"
import "core:math/rand"
import "core:time"

random_fill :: proc() -> f32 {
	return rand.float32_range(0.0, 100.0)
}

main :: proc() {
	left_size := 1024
	shared_size := 1024
	right_size := 1024
	a := make_2d_slice(shared_size, left_size, f32)
	b := make_2d_slice(right_size, shared_size, f32)
	c := make_2d_slice(right_size, left_size, f32)
	for i in 0 ..< shared_size {
		for j in 0 ..< left_size {
			a[i][j] = random_fill()
		}
	}
	for i in 0 ..< right_size {
		for j in 0 ..< shared_size {
			b[i][j] = random_fill()
		}
	}
	ma, mb, mc := new_matrix(a), new_matrix(b), new_matrix(c)
	start := time.now()
	matmul(ma, mb, mc)
	end := time.since(start)
	fmt.println(left_size, shared_size, right_size, end)
}
