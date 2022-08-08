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
kc :: 128
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
matmul :: proc(am, bm, cm: ^Matrix) #no_bounds_check{
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
	c0 := simd.f32x4{} // 4x4 row 1
	c1 := simd.f32x4{} // 4x4 row 2
	c2 := simd.f32x4{} // ...
	c3 := simd.f32x4{}
	for i in 0 ..< kc {
		ac0 := simd.f32x4{a[i][0], a[i][0], a[i][0], a[i][0]}
		ac1 := simd.f32x4{a[i][1], a[i][1], a[i][1], a[i][1]}
		ac2 := simd.f32x4{a[i][2], a[i][2], a[i][2], a[i][2]}
		ac3 := simd.f32x4{a[i][3], a[i][3], a[i][3], a[i][3]}
		br  := simd.f32x4{b[i][0], b[i][1], b[i][2], b[i][3]}
		c0   = simd.fma(ac0, br, c0)
		c1   = simd.fma(ac1, br, c1)
		c2   = simd.fma(ac2, br, c2)
		c3   = simd.fma(ac3, br, c3)
	}
	c[0][k+0] += simd.extract(c0, 0)
	c[1][k+0] += simd.extract(c0, 1)
	c[2][k+0] += simd.extract(c0, 2)
	c[3][k+0] += simd.extract(c0, 3)
	c[0][k+1] += simd.extract(c1, 0)
	c[1][k+1] += simd.extract(c1, 1)
	c[2][k+1] += simd.extract(c1, 2)
	c[3][k+1] += simd.extract(c1, 3)
	c[0][k+2] += simd.extract(c2, 0)
	c[1][k+2] += simd.extract(c2, 1)
	c[2][k+2] += simd.extract(c2, 2)
	c[3][k+2] += simd.extract(c2, 3)
	c[0][k+3] += simd.extract(c3, 0)
	c[1][k+3] += simd.extract(c3, 1)
	c[2][k+3] += simd.extract(c3, 2)
	c[3][k+3] += simd.extract(c3, 3)

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

/* matmul :: proc(a, b, c: ^Matrix) { */
/* 	assert(a.n_cols == b.n_rows && c.n_rows == a.n_rows && c.n_cols == b.n_cols) */
/* 	m, n, k := a.n_rows, b.n_cols, a.n_cols */
/* 	outer_kernel(m, n, k, a.data, b.data, c.data) */
/* } */

/* packed_t :: distinct [kc][4]f32 */
/* import "core:time" */
/* outer_kernel :: proc(m, n, k: int, a, b, c: [][]f32) { */
/* 	// This time, we compute a mc x n block of C by a call to the InnerKernel */
/* 	/* start := time.now() */ */
/* 	packed_a := make([]packed_t, mc / 4) */
/* 	packed_b := make([]packed_t, len(b) / 4) */
/* 	/* end := time.since(start) */ */
/*  /*  fmt.println(end) */ */
/* 	for p := 0; p < k - 1; p += kc { // loop over a cols/b rows => c cols */
/* 		for i := 0; i < m - 1; i += mc { // c rows */
/* 			inner_kernel(i, p, a[p:p + kc], b, c, &packed_a, &packed_b, i == 0) */
/* 		} */
/* 	} */
/* } */

/* inner_kernel :: proc(m, k: int, a, b, c: [][]f32, packed_a, packed_b: ^[]packed_t, first_time: bool) { */
/* 	for bpc, j in packed_b { /* Loop over the columns of C, unrolled by 4 */ */
/* 		jl := j * 4 */
/* 		if first_time { */
/* 			pack_matrix_b(k, b[jl:jl + 4], &bpc) */
/* 		} */
/*     /* fmt.println(packed_b[0]) */ */
/* 		for apr, i in packed_a { /* Loop over the rows of C */ */
/* 			/* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in one routine (four inner products) */ */
/* 			if j == 0 { */
/* 				pack_matrix_a(m+i*4, a, &apr) */
/* 			} */
/*       if m>0{ */
/*         fmt.println(apr[0]) */
/*         assert(false) */
/*       } */
/*     /*   fmt.println(len(apr)) */ */
/*     //TODO: double check this k */
/* 			add_dot_4x4(k, &apr, &bpc, c[jl:jl + 4]) */
/* 		} */
/* 	} */
/* } */


/* add_dot_4x4 :: proc(k: int, a, b: ^packed_t, c: [][]f32) { */
/* 	c0 := simd.f32x4{} // 4x4 row 1 */
/* 	c1 := simd.f32x4{} // 4x4 row 2 */
/* 	c2 := simd.f32x4{} // ... */
/* 	c3 := simd.f32x4{} */
/* 	for i in 0 ..< kc { */
/* 		ac0 := simd.f32x4{a[i][0], a[i][0], a[i][0], a[i][0]} */
/* 		ac1 := simd.f32x4{a[i][1], a[i][1], a[i][1], a[i][1]} */
/* 		ac2 := simd.f32x4{a[i][2], a[i][2], a[i][2], a[i][2]} */
/* 		ac3 := simd.f32x4{a[i][3], a[i][3], a[i][3], a[i][3]} */
/* 		br := simd.f32x4{b[i][0], b[i][1], b[i][2], b[i][3]} */
/* 		c0 = simd.fma(ac0, br, c0) */
/* 		c1 = simd.fma(ac1, br, c0) */
/* 		c2 = simd.fma(ac2, br, c0) */
/* 		c3 = simd.fma(ac3, br, c0) */
/* 	} */
/* 	c[0][0] += simd.extract(c0, 0) */
/* 	c[1][0] += simd.extract(c0, 1) */
/* 	c[2][0] += simd.extract(c0, 2) */
/* 	c[3][0] += simd.extract(c0, 3) */
/* 	c[0][1] += simd.extract(c1, 0) */
/* 	c[1][1] += simd.extract(c1, 1) */
/* 	c[2][1] += simd.extract(c1, 2) */
/* 	c[3][1] += simd.extract(c1, 3) */
/* 	c[0][2] += simd.extract(c2, 0) */
/* 	c[1][2] += simd.extract(c2, 1) */
/* 	c[2][2] += simd.extract(c2, 2) */
/* 	c[3][2] += simd.extract(c2, 3) */
/* 	c[0][3] += simd.extract(c3, 0) */
/* 	c[1][3] += simd.extract(c3, 1) */
/* 	c[2][3] += simd.extract(c3, 2) */
/* 	c[3][3] += simd.extract(c3, 3) */
/* } */
