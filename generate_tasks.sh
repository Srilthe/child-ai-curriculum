#!/bin/bash
mkdir -p tasks

for i in $(seq 16 215); do
  case $((i % 10)) in
    0) desc="Write a function that sorts a list of numbers using bubble sort." ;;
    1) desc="Write a function that computes the factorial of a number recursively." ;;
    2) desc="Write a function that reads a CSV and returns the number of rows." ;;
    3) desc="Write a function that plots a histogram of a numeric list and saves it." ;;
    4) desc="Write a function that counts unique words in a text file." ;;
    5) desc="Write a function that normalizes a NumPy array to mean 0, std 1." ;;
    6) desc="Write a function that trains a logistic regression classifier and returns accuracy." ;;
    7) desc="Write a function that merges two JSON files into one dictionary." ;;
    8) desc="Write a function that fetches JSON from a URL and returns it as a dict." ;;
    9) desc="Write a function that computes the nth Fibonacci number iteratively." ;;
  esac

  cat > tasks/task${i}.json <<EOF
{
  "id": "task${i}",
  "description": "${desc}"
}
EOF
done

git add tasks/task*.json
git commit -m "Add tasks 16-215 (200 new tasks)"
git push

