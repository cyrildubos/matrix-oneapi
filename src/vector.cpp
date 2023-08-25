#include <sycl/sycl.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

constexpr std::size_t size = 3;

constexpr int minimum = 0;
constexpr int maximum = 10;

std::vector<int> random_matrix(std::size_t size, int minimum, int maximum) {
  std::vector<int> vector(size * size);

  for (auto index = 0; index < size * size; ++index)
    vector[index] = (std::rand() % (maximum - minimum)) + minimum;

  return vector;
}

std::vector<int> add_matrix(std::size_t size, sycl::queue queue,
                            std::vector<int>& vector_0,
                            std::vector<int>& vector_1) {
  std::vector<int> vector_2(size * size);

  oneapi::dpl::for_each(
      oneapi::dpl::execution::make_device_policy(queue),
      oneapi::dpl::make_zip_iterator(vector_0.begin(), vector_1.begin(),
                                     vector_2.begin()),
      oneapi::dpl::make_zip_iterator(vector_0.end(), vector_1.end(),
                                     vector_2.end()),
      [](auto item) {
        std::get<2>(item) = std::get<0>(item) + std::get<1>(item);
      });

  return vector_2;
}

std::vector<int> multiply_matrix(std::size_t size, std::vector<int> vector_0,
                                 std::vector<int> vector_1, sycl::queue queue) {
  std::vector<int> vector_2(size * size);

  sycl::buffer buffer_0(vector_0);
  sycl::buffer buffer_1(vector_1);
  sycl::buffer buffer_2(vector_2);

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_0(buffer_0, handler, sycl::read_only);
    sycl::accessor accessor_1(buffer_1, handler, sycl::read_only);
    sycl::accessor accessor_2(buffer_2, handler, sycl::write_only);

    sycl::range<2> range(size, size);

    sycl::local_accessor<int, 2> local_0(range, handler);
    sycl::local_accessor<int, 2> local_1(range, handler);

    handler.parallel_for(sycl::nd_range<2>(range, range), [=](auto item) {
      auto index_0 = item.get_global_id(0);
      auto index_2 = item.get_global_id(1);

      local_0[index_0][index_2] = accessor_0[index_0 * size + index_2];
      local_1[index_0][index_2] = accessor_0[index_0 * size + index_2];

      sycl::group_barrier(item.get_group());

      for (auto index_1 = 0; index_1 < size; ++index_1)
        accessor_2[index_0 * size + index_2] +=
            accessor_0[index_0 * size + index_1] *
            accessor_1[index_1 * size + index_2];
    });
  });

  return vector_2;
}

void print_matrix(std::vector<int> vector, std::size_t size) {
  for (auto index_0 = 0; index_0 < size; ++index_0) {
    for (auto index_1 = 0; index_1 < size; ++index_1)
      std::cout << vector[index_0 * size + index_1] << ' ';

    std::cout << '\n';
  }
}

int main() {
  std::srand(std::time(nullptr));

  sycl::queue queue;

  auto vector_0 = random_matrix(size, minimum, maximum);
  auto vector_1 = random_matrix(size, minimum, maximum);

  print_matrix(vector_0, size);
  print_matrix(vector_1, size);

  auto vector_2 = add_matrix(size, queue, vector_0, vector_1);

  print_matrix(vector_2, size);

  auto vector_3 = random_matrix(size, minimum, maximum);

  print_matrix(vector_3, size);

  auto vector_4 = multiply_matrix(size, vector_2, vector_3, queue);

  print_matrix(vector_4, size);

  return 0;
}
