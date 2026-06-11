test('level1',
  executable(
    '@0@-level1-tester'.format(meson.project_name()),
    sources: files('blas/level1.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp',
  ),
)

test('level2',
  executable(
    '@0@-level2-tester'.format(meson.project_name()),
    sources: files('blas/level2.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp',
  ),
)

test('level3',
  executable(
    '@0@-level3-tester'.format(meson.project_name()),
    sources: files('blas/level3.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp',
  ),
)

test('additional-level-2',
  executable(
    '@0@-add-level2-tester'.format(meson.project_name()),
    sources: files('blas/additional-level2.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp', 
  ),
  )

test('additional-level-3',
  executable(
    '@0@-add-level3-tester'.format(meson.project_name()),
    sources: files('blas/additional-level3.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp', 
  ),
  )

test('blockdiagmatrix',
  executable(
    '@0@-blockdiagmatrix-tester'.format(meson.project_name()),
    sources: files('tensor/blockdiagmatrix.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp',
  ),
)

if has_gpu == true
  test('level1-gpu',
  executable(
    '@0@-level1-gpu-tester'.format(meson.project_name()),
    sources: files('blas/level1-gpu.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp', 
  ),
  )
  test('level2-gpu',
  executable(
    '@0@-level2-gpu-tester'.format(meson.project_name()),
    sources: files('blas/level2-gpu.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp', 
  ),
  )

  test('additional-level-1-gpu',
  executable(
    '@0@-add-level1-tester-gpu'.format(meson.project_name()),
    sources: files('blas/additional-level1-gpu.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp', 
  ),
  )

  test('additional-level-2-gpu',
  executable(
    '@0@-add-level2-tester-gpu'.format(meson.project_name()),
    sources: files('blas/additional-level2-gpu.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp', 
  ),
  )

  test('additional-level-3-gpu',
  executable(
    '@0@-add-level3-tester-gpu'.format(meson.project_name()),
    sources: files('blas/additional-level3-gpu.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp',
  ),
  )

  test('gpu-misc',
  executable(
    '@0@-misc-gpu'.format(meson.project_name()),
    sources: files('tensor/gpu-misc.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp', 
  ),
  )

  test('mat-attributes',
  executable(
    '@0@-mat-attributes'.format(meson.project_name()),
    sources: files('tensor/mat-attributes.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp', 
  ),
  )

  test('test-mp',
  executable(
    '@0@-test-mp'.format(meson.project_name()),
    sources: files('tensor/test-mp.cpp','utils.cpp'),
    dependencies: lahva_dep,
    link_language: 'cpp',
  ),
    timeout: 60 # s
  )
endif