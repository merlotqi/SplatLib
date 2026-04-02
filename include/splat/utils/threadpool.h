/**
 * @file splat/utils/threadpool.h
 * @brief Fixed-size worker pool with a task queue (internal / tooling use).
 *
 */
 
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

/** @brief Thread pool: workers drain a shared FIFO of std::function tasks. */
class ThreadPool {
 public:
  /** @param num_threads Worker threads (default: hardware_concurrency). */
  explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
              return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
          }

          task();
        }
      });
    }
  }

  /** @brief Stop workers and join threads. */
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    condition_.notify_all();

    for (auto &worker : workers_) {
      worker.join();
    }
  }

  /** @brief Run f(args...) on a worker; future holds result or exception. */
  template <typename F, typename... Args>
  auto enqueue(F &&f, Args &&...args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task =
        std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }
      tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return res;
  }

  /** @brief Pending tasks (locks mutex). */
  size_t getQueueSize() const {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return tasks_.size();
  }

  /** @brief Number of worker threads. */
  size_t getWorkerCount() const { return workers_.size(); }

  /** @brief Discard queued tasks without running. */
  void cleanupQueue() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    std::queue<std::function<void()>> empty;
    std::swap(tasks_, empty);
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  mutable std::mutex queue_mutex_;
  std::condition_variable condition_;
  std::atomic<bool> stop_;
};
