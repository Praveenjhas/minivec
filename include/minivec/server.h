#pragma once

#include "minivec/collection.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace minivec {

class Server {
public:
    Server();

    void run(int port);

private:
    void load_collections();
    void save_collection(const std::string& name,
                         const Collection& collection);

    std::unordered_map<std::string, std::unique_ptr<Collection>> collections_;
    std::mutex collections_mutex_;

    std::string data_directory_;
};

}