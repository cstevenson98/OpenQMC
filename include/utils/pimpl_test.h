#include <memory>
class MyClass {
    public:
        MyClass(int size);
        ~MyClass();
        void doSomething();
    private:
        class Impl;
        std::unique_ptr<Impl> pimpl;
};