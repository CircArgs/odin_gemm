package matmul

import "core:simd"
import "core:fmt"
// References:
// https://github.com/flame/how-to-optimize-gemm
// https://github.com/flame/blislab/blob/master/tutorial.pdf

 //Thanks to Phil H from odin discord
make_2d_slice :: proc(y, x: int, $T: typeid, allocator := context.allocator) -> (res: [][]T) {
	assert(x > 0 && y > 0)
	context.allocator = allocator

	backing := make([]T, x * y)
	res = make([][]T, y)

	for i in 0 ..< y {
		res[i] = backing[x * i:][:x]
	}
	return
}

delete_2d_slice :: proc(slice: [][]$T, allocator := context.allocator) {
	delete(slice[0], allocator)
	delete(slice, allocator)
}
/* Block sizes */
mc :: 256
kc :: 256
nc :: 1024

// column major matrix
Matrix :: struct {
	n_rows, n_cols: int,
	data:           [][]f32,
}

new_matrix :: proc(data: [][]f32) -> ^Matrix {
	ret := new(Matrix)
	ret.n_rows = len(data[0])
	ret.n_cols = len(data)
	ret.data = data
	return ret
}

packed_t :: distinct [kc][4]f32
matmul :: proc(am, bm, cm: ^Matrix){
	assert(am.n_cols == bm.n_rows && cm.n_rows == am.n_rows && cm.n_cols == bm.n_cols)
	m, n, k := am.n_rows, bm.n_cols, am.n_cols
  a, b, c:=am.data, bm.data, cm.data
  packed_a := make([]packed_t, mc / 4)
  packed_b := make([]packed_t, nc / 4)
	for jc:=0; jc<n-1; jc+=nc{
    for pc:=0; pc<k-1; pc+=kc{
      pack_matrix_b(pc, b[jc:jc+nc], &packed_b)
      for ic:=0; ic<m-1; ic+=mc{
        pack_matrix_a(ic, a[pc:pc+kc], &packed_a)
        for bp, jr in packed_b{
          for ap, ir in packed_a{
            add_dot_4x4(ic+ir*4, ap, bp, c[jc+jr*4:jc+jr*4+4])
          }
        }
      }
    }
  }
}

add_dot_4x4 :: proc(k: int, a, b: packed_t, c: [][]f32) {
	c0 := [4]f32{} // 4x4 row 1
	c1 := [4]f32{} // 4x4 row 2
	c2 := [4]f32{} // ...
	c3 := [4]f32{}
  for i in 0 ..< kc {
		ac0 := [4]f32{a[i][0], a[i][0], a[i][0], a[i][0]}
		ac1 := [4]f32{a[i][1], a[i][1], a[i][1], a[i][1]}
		ac2 := [4]f32{a[i][2], a[i][2], a[i][2], a[i][2]}
		ac3 := [4]f32{a[i][3], a[i][3], a[i][3], a[i][3]}
		br  := [4]f32{b[i][0], b[i][1], b[i][2], b[i][3]}
		c0  += ac0*br
		c1  += ac1*br
		c2  += ac2*br
		c3  += ac3*br
	}
	c[0][k+0] += c0[0]
	c[1][k+0] += c0[1]
	c[2][k+0] += c0[2]
	c[3][k+0] += c0[3]
	c[0][k+1] += c1[0]
	c[1][k+1] += c1[1]
	c[2][k+1] += c1[2]
	c[3][k+1] += c1[3]
	c[0][k+2] += c2[0]
	c[1][k+2] += c2[1]
	c[2][k+2] += c2[2]
	c[3][k+2] += c2[3]
	c[0][k+3] += c3[0]
	c[1][k+3] += c3[1]
	c[2][k+3] += c3[2]
	c[3][k+3] += c3[3]
}
pack_matrix_a :: proc(m: int, a: [][]f32, a_to: ^[]packed_t) {
  for ap, i in a_to{
    for ac, j in &ap {
      ac[0] = a[j][m+i*4 + 0]
      ac[1] = a[j][m+i*4 + 1]
      ac[2] = a[j][m+i*4 + 2]
      ac[3] = a[j][m+i*4 + 3]
    }
  }
}

pack_matrix_b :: proc(k: int, b: [][]f32, b_to: ^[]packed_t) {
  for bp, i in b_to{
    b_i0_pntr := b[i*4+0][k:]
    b_i1_pntr := b[i*4+1][k:]
    b_i2_pntr := b[i*4+2][k:]
    b_i3_pntr := b[i*4+3][k:]
    for br , j in &bp{
      br[0] = b_i0_pntr[j]
      br[1] = b_i1_pntr[j]
      br[2] = b_i2_pntr[j]
      br[3] = b_i3_pntr[j]
    }
  }
}

