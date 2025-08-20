cmake_minimum_required(VERSION 3.8)
project(launch_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------
# 필수 의존성
# -----------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)  # Python launch 파일 지원

# -----------------------------
# Linter 테스트 설정 (선택)
# -----------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE) # copyright 검사 무시
  set(ament_cmake_cpplint_FOUND TRUE)   # cpplint 검사 무시
  ament_lint_auto_find_test_dependencies()
endif()

# -----------------------------
# launch 파일 설치
# -----------------------------
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# -----------------------------
# 패키지 내보내기
# -----------------------------
ament_package()

