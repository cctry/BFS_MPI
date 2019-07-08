#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include "graph.h"
#include <string>
#define ROOT 0
#define INFTY 255
using graph_t = graph<long, long, int, long, long, char>;
using depth_t = unsigned char;
struct metadata_t {
    long vert_count;
    long edge_count;
};

const std::string path = "/mnt/e/graphs/orkut-links/";
const std::string prefix = "out.orkut-links";
const std::string beg_file = path + prefix + "_beg_pos.bin";
const std::string csr_file = path + prefix + "_csr.bin";
const std::string weight_file = path + prefix + "_weight.bin";

long start = 0;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    graph_t* g = nullptr;
    metadata_t mdata;
    graph_t::vert_t *beg_pos, *csr;
    if (rank == ROOT) {  // load the graph
        g = new graph_t(beg_file.c_str(), csr_file.c_str(),
                        weight_file.c_str());
        mdata = {g->vert_count, g->edge_count};
    }
    MPI_Bcast(&mdata, sizeof(metadata_t), MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank == ROOT) {
        beg_pos = g->beg_pos;
        csr = g->csr;
    } else {
        beg_pos = new graph_t::vert_t[mdata.vert_count];
        csr = new graph_t::vert_t[mdata.edge_count];
    }
    MPI_Bcast(beg_pos, mdata.vert_count, MPI_LONG, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(csr, mdata.edge_count, MPI_LONG, ROOT, MPI_COMM_WORLD);
    auto vert_vec = new depth_t[mdata.vert_count];
    auto vert_vec_res = new depth_t[mdata.vert_count];
    memset(vert_vec, INFTY, sizeof(depth_t) * mdata.vert_count);
    vert_vec[start] = 0;
    depth_t level = 0;
    auto step = mdata.vert_count / size;
    auto vert_beg = rank * step;
    auto vert_end = ((rank == size - 1) ? mdata.vert_count : vert_beg + step);
    long* res = new long[255];
    while (true) {
        for (graph_t::vert_t vert_id = vert_beg; vert_id < vert_end;
             vert_id++) {
            if (vert_vec[vert_id] == level) {
                auto my_beg = beg_pos[vert_id];
                auto my_end = beg_pos[vert_id + 1];
                for (; my_beg < my_end; my_beg++) {
                    auto nebr = csr[my_beg];
                    if (vert_vec[nebr] == INFTY) {
                        vert_vec[nebr] = level + 1;
                    }
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(vert_vec, vert_vec_res, mdata.vert_count, MPI_UNSIGNED_CHAR,
                   MPI_MIN, ROOT, MPI_COMM_WORLD);
        if (rank == ROOT) {
            memcpy(vert_vec, vert_vec_res, sizeof(depth_t) * mdata.vert_count);
        }
        MPI_Bcast(vert_vec, mdata.vert_count, MPI_UNSIGNED_CHAR, ROOT,
                  MPI_COMM_WORLD);
        // result and exit
        memset(res, 0, sizeof(long) * 255);
        int flag = 0;
        for (graph_t::vert_t vert_id = 0; vert_id < mdata.vert_count;
             vert_id++) {
            auto temp = vert_vec[vert_id];
            if (temp == INFTY) {
                flag = 1;
                break;
            }
            res[temp]++;
        }
        if (flag == 0) {
            if (rank == ROOT) {
                for (int i = 1; i < 255; i++) {
                    if (res[i] == 0)
                        break;
                    else {
                        printf("Level-%d-data-meta-total: %ld\n", i, res[i]);
                    }
                }
            }
            break;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        level++;
    }
    MPI_Finalize();
    return 0;
}