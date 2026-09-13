#include "minivec/server.h"

int main() {
    minivec::Server server;
    server.run(18080);

    return 0;
}