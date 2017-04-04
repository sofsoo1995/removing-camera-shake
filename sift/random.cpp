#include<iostream>
#include<random>

using namespace std;

int main(int argc, char *argv[]){
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> uni(0,25);
  for (int i = 0; i < 100; i++) {
    auto random_integer = uni(rng);
    cout<< random_integer<<endl;
  }
  return 0;
}

