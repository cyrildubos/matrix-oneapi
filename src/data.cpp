#include <sycl/sycl.hpp>

constexpr std::size_t size_0 = 3;
constexpr std::size_t size_1 = 6;
constexpr std::size_t size_2 = 4;

constexpr int minimum = 0;
constexpr int maximum = 10;

int* random_matrix(std::size_t size_0, std::size_t size_1, sycl::queue queue,
                   int minimum, int maximum) {
  auto data = sycl::malloc_shared<int>(size_0 * size_1, queue);

  for (auto index = 0; index < size_0 * size_1; ++index)
    data[index] = (std::rand() % (maximum - minimum)) + minimum;

  return data;
}

int* add_matrix(std::size_t size_0, std::size_t size_1, sycl::queue queue,
                int* data_0, int* data_1) {
  auto data_2 = sycl::malloc_shared<int>(size_0 * size_1, queue);

  queue.parallel_for(sycl::range<2>(size_0, size_1), [=](auto indices) {
    auto index_0 = indices[0];
    auto index_1 = indices[1];

    data_2[index_0 * size_1 + index_1] =
        data_0[index_0 * size_1 + index_1] + data_1[index_0 * size_1 + index_1];
  });

  return data_2;
}

int* multiply_matrix(std::size_t size_0, std::size_t size_1, std::size_t size_2,
                     sycl::queue queue, int* data_0, int* data_1) {
  auto data_2 = sycl::malloc_shared<int>(size_0 * size_2, queue);

  queue.parallel_for(sycl::range<2>(size_0, size_2), [=](auto indices) {
    auto index_0 = indices[0];
    auto index_2 = indices[1];

    data_2[index_0 * size_2 + index_2] = 0;

    for (auto index_1 = 0; index_1 < size_1; ++index_1)
      data_2[index_0 * size_2 + index_2] += data_0[index_0 * size_1 + index_1] *
                                            data_1[index_1 * size_2 + index_2];
  });

  return data_2;
}

void print_matrix(int* data, std::size_t size_0, std::size_t size_1) {
  for (auto index_0 = 0; index_0 < size_0; ++index_0) {
    for (auto index_1 = 0; index_1 < size_1; ++index_1)
      std::cout << data[index_0 * size_1 + index_1] << ' ';

    std::cout << '\n';
  }
}

int main() {
  std::srand(std::time(nullptr));

  sycl::queue queue;

  auto data_0 = random_matrix(size_0, size_1, queue, minimum, maximum);
  auto data_1 = random_matrix(size_0, size_1, queue, minimum, maximum);

  queue.wait();

  print_matrix(data_0, size_0, size_1);
  print_matrix(data_1, size_0, size_1);

  auto data_2 = add_matrix(size_0, size_1, queue, data_0, data_1);

  queue.wait();

  print_matrix(data_2, size_0, size_1);

  auto data_3 = random_matrix(size_1, size_2, queue, minimum, maximum);

  queue.wait();

  print_matrix(data_3, size_1, size_2);

  auto data_4 = multiply_matrix(size_0, size_1, size_2, queue, data_2, data_3);

  queue.wait();

  print_matrix(data_4, size_0, size_2);

  return 0;
}