
#include "application.h"

int main(int argc, char** argv) {
  try {
    Application app(1600, 900);
    return app.run();

  } catch (const std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
