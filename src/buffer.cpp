#include <sycl/sycl.hpp>

constexpr std::size_t size_0 = 3;
constexpr std::size_t size_1 = 6;
constexpr std::size_t size_2 = 4;

constexpr int minimum = 0;
constexpr int maximum = 10;

sycl::buffer<int, 2> random_matrix(std::size_t size_0, std::size_t size_1,
                                   int minimum, int maximum) {
  sycl::buffer<int, 2> buffer(sycl::range<2>(size_0, size_1));

  sycl::host_accessor accessor(buffer, sycl::write_only);

  for (auto index_0 = 0; index_0 < size_0; ++index_0)
    for (auto index_1 = 0; index_1 < size_1; ++index_1)
      accessor[index_0][index_1] =
          (std::rand() % (maximum - minimum)) + minimum;

  return buffer;
}

sycl::buffer<int, 2> add_matrix(std::size_t size_0, std::size_t size_1,
                                sycl::queue queue,
                                sycl::buffer<int, 2> buffer_0,
                                sycl::buffer<int, 2> buffer_1) {
  sycl::buffer<int, 2> buffer_2(sycl::range<2>(size_0, size_1));

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_0(buffer_0, handler, sycl::read_only);
    sycl::accessor accessor_1(buffer_1, handler, sycl::read_only);
    sycl::accessor accessor_2(buffer_2, handler, sycl::write_only);

    handler.parallel_for(sycl::range<2>(size_0, size_1), [=](auto indices) {
      auto index_0 = indices[0];
      auto index_1 = indices[1];

      accessor_2[index_0][index_1] =
          accessor_0[index_0][index_1] + accessor_1[index_0][index_1];
    });
  });

  return buffer_2;
}

sycl::buffer<int, 2> multiply_matrix(std::size_t size_0, std::size_t size_1,
                                     std::size_t size_2, sycl::queue queue,
                                     sycl::buffer<int, 2> buffer_0,
                                     sycl::buffer<int, 2> buffer_1) {
  sycl::buffer<int, 2> buffer_2(sycl::range<2>(size_0, size_2));

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_0(buffer_0, handler, sycl::read_only);
    sycl::accessor accessor_1(buffer_1, handler, sycl::read_only);
    sycl::accessor accessor_2(buffer_2, handler, sycl::write_only);

    handler.parallel_for(sycl::range<2>(size_0, size_2), [=](auto indices) {
      auto index_0 = indices[0];
      auto index_2 = indices[1];

      accessor_2[index_0][index_2] = 0;

      for (auto index_1 = 0; index_1 < size_1; ++index_1)
        accessor_2[index_0][index_2] +=
            accessor_0[index_0][index_1] * accessor_1[index_1][index_2];
    });
  });

  return buffer_2;
}

void print_matrix(sycl::buffer<int, 2> buffer, std::size_t size_0,
                  std::size_t size_1) {
  sycl::host_accessor accessor(buffer, sycl::read_only);

  for (auto index_0 = 0; index_0 < size_0; ++index_0) {
    for (auto index_1 = 0; index_1 < size_1; ++index_1)
      std::cout << accessor[index_0][index_1] << ' ';

    std::cout << '\n';
  }
}

int main() {
  std::srand(std::time(nullptr));

  sycl::queue queue;

  auto buffer_0 = random_matrix(size_0, size_1, minimum, maximum);
  auto buffer_1 = random_matrix(size_0, size_1, minimum, maximum);

  print_matrix(buffer_0, size_0, size_1);
  print_matrix(buffer_1, size_0, size_1);

  auto buffer_2 = add_matrix(size_0, size_1, queue, buffer_0, buffer_1);

  print_matrix(buffer_2, size_0, size_1);

  auto buffer_3 = random_matrix(size_1, size_2, minimum, maximum);

  print_matrix(buffer_3, size_1, size_2);

  auto buffer_4 =
      multiply_matrix(size_0, size_1, size_2, queue, buffer_2, buffer_3);

  print_matrix(buffer_4, size_0, size_2);
}
