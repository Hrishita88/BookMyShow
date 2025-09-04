{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyM4J6gIjL+uVlTPoPOy9I9A",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Hrishita88/BookMyShow/blob/main/week1_activity.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "<h1>Part 1: Vector Operations</h1>"
      ],
      "metadata": {
        "id": "hWZiPqgisLjH"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 55,
      "metadata": {
        "id": "B1CbQ8uwlXVZ"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import time"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def dot_product_loop(vec1, vec2):\n",
        "    \"\"\"\n",
        "    Calculate dot product using a for loop.\n",
        "\n",
        "    Parameters:\n",
        "    vec1 (list): First vector as a Python list\n",
        "    vec2 (list): Second vector as a Python list\n",
        "\n",
        "    Returns:\n",
        "    float: The dot product of vec1 and vec2\n",
        "    \"\"\"\n",
        "    result = 0\n",
        "    for i in range(len(vec1)):\n",
        "      result+= vec1[i] * vec2[i]\n",
        "    return result\n",
        "    pass\n",
        "\n",
        "# vec1= ([1, 2, 3, 4])\n",
        "# vec2= ([6, 7, 8, 8])\n",
        "\n",
        "# print(dot_product_loop(vec1, vec2))"
      ],
      "metadata": {
        "id": "fYwHlA16lhil"
      },
      "execution_count": 56,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def dot_product_numpy(vec1, vec2):\n",
        "    \"\"\"\n",
        "    Calculate dot product using NumPy.\n",
        "\n",
        "    Parameters:\n",
        "    vec1 (np.ndarray): First vector as NumPy array\n",
        "    vec2 (np.ndarray): Second vector as NumPy array\n",
        "\n",
        "    Returns:\n",
        "    float: The dot product of vec1 and vec2\n",
        "    \"\"\"\n",
        "    return vec1 @ vec2\n",
        "    pass"
      ],
      "metadata": {
        "id": "Du9cmVtuo1vN"
      },
      "execution_count": 57,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import math\n",
        "def vector_norm_loop(vec):\n",
        "    \"\"\"\n",
        "    Calculate L2 norm using a for loop.\n",
        "\n",
        "    Parameters:\n",
        "    vec (list): Input vector as a Python list\n",
        "\n",
        "    Returns:\n",
        "    float: The L2 norm of the vector\n",
        "    \"\"\"\n",
        "    result = 0\n",
        "    for i in vec:\n",
        "      result += vec**2\n",
        "    return math.sqrt(result)\n",
        "    pass"
      ],
      "metadata": {
        "id": "0lYr5o_JpVVl"
      },
      "execution_count": 58,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def vector_norm_numpy(vec):\n",
        "    \"\"\"\n",
        "    Calculate L2 norm using NumPy.\n",
        "\n",
        "    Parameters:\n",
        "    vec (np.ndarray): Input vector as NumPy array\n",
        "\n",
        "    Returns:\n",
        "    float: The L2 norm of the vector\n",
        "    \"\"\"\n",
        "    return np.sqrt(np.sum(vec**2))\n",
        "    pass"
      ],
      "metadata": {
        "id": "jQ7gAzj9rAS3"
      },
      "execution_count": 59,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "<h1>Part 2: Matrix Operations</h1>"
      ],
      "metadata": {
        "id": "Blrllff2sSm6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def matrix_multiply_loop(mat1, mat2):\n",
        "    \"\"\"\n",
        "    Multiply two matrices using nested for loops.\n",
        "\n",
        "    Parameters:\n",
        "    mat1 (list of lists): First matrix as nested Python lists\n",
        "    mat2 (list of lists): Second matrix as nested Python lists\n",
        "\n",
        "    Returns:\n",
        "    list of lists: Result of matrix multiplication\n",
        "    \"\"\"\n",
        "    result = [[0]*len(mat2[0]) for _ in range(len(mat1))]\n",
        "    for i in range(len(mat1)):\n",
        "      for j in range(len(mat2[0])):\n",
        "        for k in range(len(mat2)):\n",
        "          result[i][j]=+ mat1[i][k] * mat2[k][j]\n",
        "\n",
        "    return result\n",
        "    pass"
      ],
      "metadata": {
        "id": "Kvs6410CrxBu"
      },
      "execution_count": 60,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def matrix_multiply_numpy(mat1, mat2):\n",
        "    \"\"\"\n",
        "    Multiply two matrices using NumPy.\n",
        "\n",
        "    Parameters:\n",
        "    mat1 (np.ndarray): First matrix as NumPy array\n",
        "    mat2 (np.ndarray): Second matrix as NumPy array\n",
        "\n",
        "    Returns:\n",
        "    np.ndarray: Result of matrix multiplication\n",
        "    \"\"\"\n",
        "    return mat1 @ mat2\n",
        "    pass"
      ],
      "metadata": {
        "id": "Z4XYwKtU4LAI"
      },
      "execution_count": 61,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def matrix_transpose_loop(mat):\n",
        "    \"\"\"\n",
        "    Transpose a matrix using for loops.\n",
        "\n",
        "    Parameters:\n",
        "    mat (list of lists): Input matrix as nested Python lists\n",
        "\n",
        "    Returns:\n",
        "    list of lists: Transposed matrix\n",
        "    \"\"\"\n",
        "    transpose = [[0 for _ in range(len(mat))] for _ in range(len(mat[0]))]\n",
        "\n",
        "    for i in range(len(mat)):\n",
        "      for j in range(len(mat[0])):\n",
        "        transpose[j][i] = mat[i][j]\n",
        "\n",
        "    return transpose\n",
        "    pass"
      ],
      "metadata": {
        "id": "dmEPRXdR4eNd"
      },
      "execution_count": 62,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def matrix_transpose_numpy(mat):\n",
        "    \"\"\"\n",
        "    Transpose a matrix using NumPy.\n",
        "\n",
        "    Parameters:\n",
        "    mat (np.ndarray): Input matrix as NumPy array\n",
        "\n",
        "    Returns:\n",
        "    np.ndarray: Transposed matrix\n",
        "    \"\"\"\n",
        "    return np.transpose(mat)\n",
        "    pass"
      ],
      "metadata": {
        "id": "4mcjna0uBjcY"
      },
      "execution_count": 63,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def performance_comparison(size=1000):\n",
        "    \"\"\"\n",
        "    Compare execution time between loop-based and NumPy implementations.\n",
        "\n",
        "    Parameters:\n",
        "    size (int): Size of vectors/matrices to test\n",
        "\n",
        "    Returns:\n",
        "    dict: Dictionary containing timing results with keys:\n",
        "          - 'dot_product_loop_time'\n",
        "          - 'dot_product_numpy_time'\n",
        "          - 'matrix_multiply_loop_time'\n",
        "          - 'matrix_multiply_numpy_time'\n",
        "          - 'speedup_dot_product' (numpy_time / loop_time)\n",
        "          - 'speedup_matrix_multiply' (numpy_time / loop_time)\n",
        "    \"\"\"\n",
        "    results = {}\n",
        "\n",
        "    # Generate random test data\n",
        "    vec1_list = [np.random.random() for _ in range(size)]\n",
        "    vec2_list = [np.random.random() for _ in range(size)]\n",
        "    vec1_np = np.array(vec1_list)\n",
        "    vec2_np = np.array(vec2_list)\n",
        "\n",
        "    # For matrix multiplication, use smaller size to avoid timeout\n",
        "    mat_size = min(100, size // 10)\n",
        "    mat1_list = [[np.random.random() for _ in range(mat_size)]\n",
        "                 for _ in range(mat_size)]\n",
        "    mat2_list = [[np.random.random() for _ in range(mat_size)]\n",
        "                 for _ in range(mat_size)]\n",
        "    mat1_np = np.array(mat1_list)\n",
        "    mat2_np = np.array(mat2_list)\n",
        "\n",
        "    # Time dot product operations\n",
        "    start = time.time()\n",
        "    dot_product_loop(vec1_list, vec2_list)\n",
        "    end = time.time()\n",
        "    results['dot_product_loop_time'] = end - start\n",
        "\n",
        "    # Your timing implementation here\n",
        "    start = time.time()\n",
        "    dot_product_numpy(vec1_np, vec2_np)\n",
        "    end = time.time()\n",
        "    results['dot_product_numpy_time'] = end - start\n",
        "\n",
        "    # Time matrix multiplication operations\n",
        "    start = time.time()\n",
        "    matrix_multiply_loop(mat1_list, mat2_list)\n",
        "    end = time.time()\n",
        "    results['matrix_multiply_loop_time'] = end - start\n",
        "\n",
        "    # Your timing implementation here\n",
        "    start = time.time()\n",
        "    matrix_multiply_numpy(mat1_np, mat2_np)\n",
        "    end = time.time()\n",
        "    results['matrix_multiply_numpy_time'] = end - start\n",
        "\n",
        "    # Calculate speedup ratios\n",
        "    results['speedup_dot_product'] = results['dot_product_loop_time'] / results['dot_product_numpy_time']\n",
        "\n",
        "    # Your calculation here\n",
        "    results['speedup_matrix_multiply'] = results['matrix_multiply_loop_time'] / results['matrix_multiply_numpy_time']\n",
        "\n",
        "    return results\n",
        "\n",
        "def main():\n",
        "    \"\"\"\n",
        "    Main function to demonstrate all implementations and print results.\n",
        "    \"\"\"\n",
        "    print(\"=\" * 50)\n",
        "    print(\"NumPy vs For Loops Performance Comparison\")\n",
        "    print(\"=\" * 50)\n",
        "\n",
        "    # Test with small examples for correctness\n",
        "    vec1 = [1, 2, 3]\n",
        "    vec2 = [4, 5, 6]\n",
        "    vec1_np = np.array(vec1)\n",
        "    vec2_np = np.array(vec2)\n",
        "\n",
        "    print(\"\\n--- Correctness Check ---\")\n",
        "    print(f\"Dot Product (Loop): {dot_product_loop(vec1, vec2)}\")\n",
        "    print(f\"Dot Product (NumPy): {dot_product_numpy(vec1_np, vec2_np)}\")\n",
        "\n",
        "    mat = [[1, 2], [3, 4]]\n",
        "    mat_np = np.array(mat)\n",
        "    print(f\"\\nOriginal Matrix: {mat}\")\n",
        "    print(f\"Transpose (Loop): {matrix_transpose_loop(mat)}\")\n",
        "    print(f\"Transpose (NumPy): {matrix_transpose_numpy(mat_np).tolist()}\")\n",
        "\n",
        "    # Performance comparison\n",
        "    print(\"\\n--- Performance Analysis ---\")\n",
        "    results = performance_comparison(size=1000)\n",
        "\n",
        "    print(f\"\\nDot Product:\")\n",
        "    print(f\"  Loop Time: {results['dot_product_loop_time']:.6f} seconds\")\n",
        "    print(f\"  NumPy Time: {results['dot_product_numpy_time']:.6f} seconds\")\n",
        "    print(f\"  NumPy Speedup: {results['speedup_dot_product']:.2f}x faster\")\n",
        "\n",
        "    print(f\"\\nMatrix Multiplication:\")\n",
        "    print(f\"  Loop Time: {results['matrix_multiply_loop_time']:.6f} seconds\")\n",
        "    print(f\"  NumPy Time: {results['matrix_multiply_numpy_time']:.6f} seconds\")\n",
        "    print(f\"  NumPy Speedup: {results['speedup_matrix_multiply']:.2f}x faster\")\n",
        "\n",
        "    return results\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rHE8hmNiBRxO",
        "outputId": "d900f397-d5e9-49c9-d3e0-e8d0a7108cb3"
      },
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "==================================================\n",
            "NumPy vs For Loops Performance Comparison\n",
            "==================================================\n",
            "\n",
            "--- Correctness Check ---\n",
            "Dot Product (Loop): 32\n",
            "Dot Product (NumPy): 32\n",
            "\n",
            "Original Matrix: [[1, 2], [3, 4]]\n",
            "Transpose (Loop): [[1, 3], [2, 4]]\n",
            "Transpose (NumPy): [[1, 3], [2, 4]]\n",
            "\n",
            "--- Performance Analysis ---\n",
            "\n",
            "Dot Product:\n",
            "  Loop Time: 0.000129 seconds\n",
            "  NumPy Time: 0.000036 seconds\n",
            "  NumPy Speedup: 3.64x faster\n",
            "\n",
            "Matrix Multiplication:\n",
            "  Loop Time: 0.082114 seconds\n",
            "  NumPy Time: 0.000291 seconds\n",
            "  NumPy Speedup: 282.30x faster\n"
          ]
        }
      ]
    }
  ]
}