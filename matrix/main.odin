package matmul
import "core:fmt"
import "core:math/rand"
import "core:time"

random_fill :: proc() -> f32 {
	return rand.float32_range(0.0, 100.0)
}

matmul_naive :: proc(a, b, c: [][]f32) {
	for i in 0 ..< len(a) {
		for j in 0 ..< len(b[0]) {
			for k in 0 ..< len(b) {
				c[i][j] += a[i][k] * b[k][j]
			}
		}
	}
}

main :: proc() {
	left_size :=   1024
	shared_size := 1024
	right_size :=  1024
	a := make_2d_slice(shared_size, left_size, f32)
	b := make_2d_slice(right_size, shared_size, f32)
	c := make_2d_slice(right_size, left_size, f32)
	for i in 0 ..< shared_size {
		for j in 0 ..< left_size {
			a[j][i] = min(f32(i+j)+4, 100)//random_fill()
		}
	}
	for i in 0 ..< right_size {
		for j in 0 ..< shared_size {
			b[j][i] = min(f32(i+j), 100)//random_fill()
		}
	}
	ma, mb, mc := new_matrix(a), new_matrix(b), new_matrix(c)
	start := time.now()
	matmul(ma, mb, mc)
	end := time.since(start)
	fmt.println(left_size, shared_size, right_size, end)

  gt:=make_2d_slice(right_size, left_size, f32)
	start = time.now()
  matmul_naive(a, b, gt)
  end = time.since(start)
	fmt.println(left_size, shared_size, right_size, end)
  /* fmt.println(c, "\n\n", gt) */

  for i in 0..<len(c){
    for j in 0..<len(c[0]){
      assert(abs(c[i][j]-gt[j][i])<1e-3)
    }
  }
}
