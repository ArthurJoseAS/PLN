#include <cctype>
#include <cstddef>
#include <vector>
#include <optional>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "stopwords.h"


std::map<std::string, int> vocabulario;

struct Review {
    std::map<std::string, int> word_counts;
    int label;
};

std::vector<Review> reviews;

std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> words;
    std::string current;

    for (unsigned char c : s) {
        if (std::isalpha(c)) {
            current += std::tolower(c);
        } else {
            if (!current.empty() && !stopwords.count(current)) {
                words.push_back(current);
            }
            current.clear();
        }
    }

    if (!current.empty() && !stopwords.count(current)) {
        words.push_back(current);
    }

    return words;
}

std::string strip_string(std::string s){
    std::string construct = "";
    for(auto a: s){
        if(std::isalpha(a)){
            construct+=a;
        }
    }
    return construct;
}

std::optional<std::ifstream> open_file(char* file_arg){
    std::ifstream f(file_arg);
    if(!f.is_open()){
        return {};
    }
    return f;
}

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Adicione um arquivo para abrir\n";
        return 1;
    }
    auto arquivo = open_file(argv[1]);
    if(!arquivo){
        std::cerr << "Arquivo invalido\n";
        return 1;
    }
    std::map<std::string, int> bow;
    std::string line;

    while (std::getline(arquivo.value(), line)) {
        size_t separator = line.rfind('\t');

        if (separator == std::string::npos)
            continue;

        std::string text = line.substr(0, separator);
        int label = line[separator + 1] - '0';

        Review review;
        review.label = label;

        for (const auto& word : tokenize(text)) {
            review.word_counts[word]++;
            vocabulario[word] = 1;
        }

        reviews.push_back(review);
    }
    std::ofstream output("Imdb_BOW.csv");

    for (const auto& [word, _] : vocabulario) {
        output << word << ",";
    }
    output << "label\n";

    for (const auto& review : reviews) {
        for (const auto& [word, _] : vocabulario) {
            output << review.word_counts.count(word)
                << ",";
        }

        output << review.label << "\n";
    }
}