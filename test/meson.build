test('level1',
  executable(
    '@0@-level1-tester'.format(meson.project_name()),
    sources: files('level1.cpp','utils.cpp'),
    dependencies: tcgmblas_dep,
  ),
)

test('level2',
  executable(
    '@0@-level2-tester'.format(meson.project_name()),
    sources: files('level2.cpp','utils.cpp'),
    dependencies: tcgmblas_dep,
  ),
)

if gpu == true
  test('level1-gpu',
  executable(
    '@0@-level1-gpu-tester'.format(meson.project_name()),
    sources: files('level1-gpu.cpp','utils.cpp'),
    dependencies: tcgmblas_dep,
  ),
  )
  test('level2-gpu',
  executable(
    '@0@-level2-gpu-tester'.format(meson.project_name()),
    sources: files('level2-gpu.cpp','utils.cpp'),
    dependencies: tcgmblas_dep,
  ),
  )
endif