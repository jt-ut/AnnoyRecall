#ifndef ANNOYLEARN_ANNOYRECALL_HPP
#define ANNOYLEARN_ANNOYRECALL_HPP


#include "annoy/annoylib.h"
#include "annoy/kissrandom.h"
#include <omp.h>
#include <map>

typedef double ANNOYTYPE;
typedef Annoy::AnnoyIndex <int, ANNOYTYPE, Annoy::Euclidean, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy> MyAnnoyIndex;
const std::vector<int> XL_empty = {};
typedef std::pair<int,int> Tpairint;
typedef std::map<Tpairint, int> TCADJMap;
typedef std::map<int,int> TLabelMap;

#pragma once

inline void RFL_Winner(int& WinLabel, double& Purity, const TLabelMap& x) {

  TLabelMap::const_iterator bestit;
  int bestval = 0;
  int sumval = 0;
  for(TLabelMap::const_iterator it = x.begin(); it != x.end(); ++it) {
    if(it->second > bestval) {
      bestit = it;
      bestval = it->second;
    }
    sumval += it->second;
  }

  WinLabel = bestit->first;

  Purity = 1.0 - std::sqrt(1.0 - std::sqrt(double(bestval) / double(sumval))); // 1 - Hellinger Distance(ideal, observed)

  return;
}

// *** Versions that write BMU & QE as a vector, with indices 0->(N-1) holding BMU1, elements N->(2N-1) holding BMu2, etc.
// inline void cpp_AnnoyBMU(std::vector<int>& BMU, std::vector<double>& QE,
//                           const std::vector<double>& X,
//                           const std::vector<double>& W,
//                           unsigned int d,
//                           unsigned int nBMU = 2,
//                           unsigned int nAnnoyTrees = 50) {
//
//   unsigned int N = X.size() / d;
//   unsigned int M = W.size() / d;
//
//   // Build Annoy indexing object
//   MyAnnoyIndex AnnoyObj(d);
//
//   for(unsigned int i=0; i<M; ++i) {
//     std::vector<double>::const_iterator it = W.cbegin() + i*d;
//     AnnoyObj.add_item(i, &(*it));
//   }
//
//   AnnoyObj.build(nAnnoyTrees);
//
//
//   BMU.resize(N * nBMU);
//   QE.resize(N * nBMU);
//
//   // Find BMU of each x
//   #pragma omp parallel for
//   for(unsigned int i=0; i<N; ++i) {
//
//     std::vector<int> tmp_BMU;
//     std::vector<double> tmp_QE;
//
//     std::vector<double>::const_iterator it = X.cbegin() + i*d;
//     AnnoyObj.get_nns_by_vector(&(*it), nBMU, -1, &tmp_BMU, &tmp_QE);
//     for(unsigned int j=0; j<nBMU; ++j) {
//       BMU[i + j*N] = tmp_BMU[j];
//       QE[i + j*N] = tmp_QE[j];
//     }
//
//     // ** The following SHOULD replace the j loop above, but std::copy_n doesn't seem to work in parallel
//     //std::copy_n(tmp_BMU.begin() + i*nBMU, nBMU, BMU.begin() + i*nBMU);
//     //std::copy_n(tmp_QE.begin() + i*nBMU, nBMU, QE.begin() + i*nBMU);
//   }
//
//   return;
// }
inline void cpp_AnnoyBMU(std::vector<int>& BMU, std::vector<double>& QE,
                          const std::vector<double>& X,
                          const std::vector<double>& W,
                          unsigned int d,
                          unsigned int nBMU = 2,
                          unsigned int nAnnoyTrees = 50) {

  //unsigned int N, M;

  // Find & set N = num. data vectors, M = num. codebook vectors
  unsigned int N = (X.size() % d==0) ? X.size() / d : throw std::runtime_error("len(X) % d != 0");
  unsigned int M = (W.size() % d==0) ? W.size() / d : throw std::runtime_error("len(W) % d != 0");

  // Build Annoy indexing object
  MyAnnoyIndex AnnoyObj(d);

  for(unsigned int i=0; i<M; ++i) {
    std::vector<double>::const_iterator it = W.cbegin() + i*d;
    AnnoyObj.add_item(i, &(*it));
  }

  AnnoyObj.build(nAnnoyTrees);


  BMU.resize(N * nBMU);
  QE.resize(N * nBMU);

  // Find BMU of each x
  #pragma omp parallel for
  for(unsigned int i=0; i<N; ++i) {

    std::vector<int> tmp_BMU;
    std::vector<double> tmp_QE;

    std::vector<double>::const_iterator it = X.cbegin() + i*d;
    AnnoyObj.get_nns_by_vector(&(*it), nBMU, -1, &tmp_BMU, &tmp_QE);
    for(unsigned int j=0; j<nBMU; ++j) {
      BMU[i + j*N] = tmp_BMU[j];
      QE[i + j*N] = tmp_QE[j];
    }

    // ** The following SHOULD replace the j loop above, but std::copy_n doesn't seem to work in parallel
    //std::copy_n(tmp_BMU.begin() + i*nBMU, nBMU, BMU.begin() + i*nBMU);
    //std::copy_n(tmp_QE.begin() + i*nBMU, nBMU, QE.begin() + i*nBMU);
  }

  return;
}

inline void cpp_AnnoyRecall(std::vector<std::vector<int>>& RF,
                            std::vector<int>& RF_Size,
                            std::vector<int>& CADJi, std::vector<int>& CADJj, std::vector<int>& CADJx,
                            std::vector<TLabelMap>& RFL_Dist,
                            std::vector<int>& RFL,
                            std::vector<double>& RFL_Purity, double& RFL_Purity_UOA, double& RFL_Purity_WOA,
                            const std::vector<int>& BMU,
                            const std::vector<double>& QE,
                            unsigned int N,
                            unsigned int M,
                            const std::vector<int>& XL = XL_empty
                            ) {
  // BMU & QE should be arrange such that elements 0->N-1 represent BMU1 of each X, elements N->2N-1 represent BMU2 of each X, and so on
  // BMU must have length >= 2N to calculate CADJ

  RF.resize(M);
  RF_Size.resize(M);
  std::fill(RF_Size.begin(), RF_Size.end(), 0);

  std::vector<int> XL_unq;
  unsigned int XL_N_unq = 0;
  TLabelMap XLMap;
  if(XL.size() > 0) {
    XL_unq = XL;
    std::vector<int>::iterator last = std::unique(XL_unq.begin(), XL_unq.end());
    XL_unq.erase(last, XL_unq.end());
    std::sort(XL_unq.begin(), XL_unq.end());
    XL_N_unq = XL_unq.size();

    for(unsigned int i=0; i<XL_N_unq; ++i) {
      XLMap[XL_unq[i]] = 0;
    }

    RFL_Dist.resize(M);
    std::fill(RFL_Dist.begin(), RFL_Dist.end(), XLMap);
    RFL.resize(M);
    std::fill(RFL.begin(), RFL.end(), 0);
    RFL_Purity.resize(M);
    std::fill(RFL_Purity.begin(), RFL_Purity.end(), 0.0);
  }

  double purWOA_num = 0.0, purWOA_denom = 0.0, purUOA_num = 0.0, purUOA_denom = 0.0;
  #pragma omp declare reduction (mergevec : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
  #pragma omp parallel for reduction(mergevec: CADJi) reduction(mergevec: CADJj) reduction(mergevec:CADJx) reduction(+:purWOA_num) reduction(+:purWOA_denom) reduction(+:purUOA_num) reduction(+:purUOA_denom)
  for(unsigned int i=0; i<M; ++i) {

    TCADJMap mapCADJ;

    for(unsigned int j=0; j<N; ++j) {
      if(BMU[j]==i) {
        RF[i].push_back(j);
        RF_Size[i]++;
        mapCADJ[std::make_pair(i, BMU[j + N])]++;
        if(XL_N_unq > 0) RFL_Dist[i][XL[j]]++; // Add labels, if they exist
      }
    }

    for(TCADJMap::iterator it = mapCADJ.begin(); it != mapCADJ.end(); ++it) {
      CADJi.push_back(it->first.first);
      CADJj.push_back(it->first.second);
      CADJx.push_back(it->second);
    }

    if(XL_N_unq > 0 && RF_Size[i] > 0) {
      RFL_Winner(RFL[i], RFL_Purity[i], RFL_Dist[i]);

      // Update overall average purities
      purWOA_num += double(RF_Size[i]) * RFL_Purity[i];
      purWOA_denom += double(RF_Size[i]);
      purUOA_num += RFL_Purity[i];
      purUOA_denom += 1.0;
    }

  }

  RFL_Purity_UOA = (purUOA_denom > 0.0) ? purUOA_num / purUOA_denom : 0.0;
  RFL_Purity_WOA = (purWOA_denom > 0.0) ? purWOA_num / purWOA_denom : 0.0;

  return;
}



// inline void cpp_AnnoyRecall(std::vector<std::vector<int>>& RF,
//                             std::vector<int>& RF_Size,
//                             std::vector<int>& CADJi, std::vector<int>& CADJj, std::vector<int>& CADJx,
//                             std::vector<TLabelMap>& RFL_Dist,
//                             std::vector<int>& RFL,
//                             std::vector<double>& RFL_Purity,
//                             const std::vector<int>& BMU,
//                             const std::vector<double>& QE,
//                             unsigned int N,
//                             unsigned int M,
//                             const std::vector<int>& XL = XL_empty
// ) {
//   // BMU & QE should be arrange such that elements 0->N-1 represent BMU1 of each X, elements N->2N-1 represent BMU2 of each X, and so on
//   // BMU must have length >= 2N to calculate CADJ
//
//   RF.resize(M);
//   RF_Size.resize(M);
//   std::fill(RF_Size.begin(), RF_Size.end(), 0);
//
//   std::vector<int> XL_unq;
//   unsigned int XL_N_unq = 0;
//   TLabelMap XLMap;
//   if(XL.size() > 0) {
//     XL_unq = XL;
//     std::vector<int>::iterator last = std::unique(XL_unq.begin(), XL_unq.end());
//     XL_unq.erase(last, XL_unq.end());
//     std::sort(XL_unq.begin(), XL_unq.end());
//     XL_N_unq = XL_unq.size();
//
//     for(unsigned int i=0; i<XL_N_unq; ++i) {
//       XLMap[XL_unq[i]] = 0;
//     }
//
//     RFL_Dist.resize(M);
//     std::fill(RFL_Dist.begin(), RFL_Dist.end(), XLMap);
//     RFL.resize(M);
//     std::fill(RFL.begin(), RFL.end(), 0);
//     RFL_Purity.resize(M);
//     std::fill(RFL_Purity.begin(), RFL_Purity.end(), 0.0);
//   }
//
//
// #pragma omp declare reduction (mergevec : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
// #pragma omp parallel for reduction(mergevec: CADJi) reduction(mergevec: CADJj) reduction(mergevec:CADJx)
//   for(unsigned int i=0; i<M; ++i) {
//
//     TCADJMap mapCADJ;
//
//     for(unsigned int j=0; j<N; ++j) {
//       if(BMU[j]==i) {
//         RF[i].push_back(j);
//         RF_Size[i]++;
//         mapCADJ[std::make_pair(i, BMU[j + N])]++;
//         if(XL_N_unq > 0) RFL_Dist[i][XL[j]]++; // Add labels, if they exist
//       }
//     }
//
//     for(TCADJMap::iterator it = mapCADJ.begin(); it != mapCADJ.end(); ++it) {
//       CADJi.push_back(it->first.first);
//       CADJj.push_back(it->first.second);
//       CADJx.push_back(it->second);
//     }
//
//     if(XL_N_unq > 0 && RFSize[i] > 0) {
//       RFL_Winner(RFL[i], RFL_Purity[i], RFL_Dist[i]);
//     }
//
//   }
//
//   return;
// }


// *** Versions that write BMU & QE as a list, with BMU[0] holding c(BMU1,BMU2,...) of X[0], BMU[1] holding c(BMU1,BMU2,...) of X[1], etc.
inline void cpp_AnnoyBMU(std::vector<std::vector<int>>& BMU,
                         std::vector<std::vector<double>>& QE,
                         const std::vector<double>& X,
                         const std::vector<double>& W,
                         unsigned int d,
                         unsigned int nBMU = 2,
                         unsigned int nAnnoyTrees = 50) {

  unsigned int N = X.size() / d;
  unsigned int M = W.size() / d;

  // Build Annoy indexing object
  MyAnnoyIndex AnnoyObj(d);

  for(unsigned int i=0; i<M; ++i) {
    std::vector<double>::const_iterator it = W.cbegin() + i*d;
    AnnoyObj.add_item(i, &(*it));
  }

  AnnoyObj.build(nAnnoyTrees);


  BMU.resize(N);
  QE.resize(N);

  // Find BMU of each x
#pragma omp parallel for
  for(unsigned int i=0; i<N; ++i) {

    std::vector<double>::const_iterator it = X.cbegin() + i*d;
    AnnoyObj.get_nns_by_vector(&(*it), nBMU, -1, &BMU[i], &QE[i]);
  }

  return;
}


class VQRecall {
public:
  int d;
  int nBMU;
  int nAnnoyTrees;

  int N;
  int M;
  std::vector<int> BMU;
  std::vector<double> QE;

  std::vector<std::vector<int>> RF;
  std::vector<int> RF_Size;
  std::vector<int> CADJi, CADJj, CADJ;

  std::vector<TLabelMap> RFL_Dist;
  std::vector<int> RFL;
  std::vector<double> RFL_Purity;
  double RFL_Purity_UOA;
  double RFL_Purity_WOA;

  // Constructor
  VQRecall(int d, int nBMU, int nAnnoyTrees) : d(d), nBMU(nBMU), nAnnoyTrees(nAnnoyTrees) {
    if(nBMU < 2) {
      throw std::runtime_error("nBMU must be >= 2");
    }

    this->flag_BMU = false;
    this->flag_Recall = false;
    this->flag_RecallLabels = false;
  };

  // set BMUs & QE
  void calc_BMU(const std::vector<double>& X, const std::vector<double>& W);

  // Full Recall
  void Recall(const std::vector<double>& X, const std::vector<double>& W, const std::vector<int>& XL = XL_empty);

private:
  bool flag_BMU;
  bool flag_Recall;
  bool flag_RecallLabels;
};

inline void VQRecall::calc_BMU(const std::vector<double>& X, const std::vector<double>& W) {
  // Find & set N = num. data vectors, M = num. codebook vectors
  this->N = (X.size() % d==0) ? X.size() / d : throw std::runtime_error("len(X) % d != 0");
  this->M = (W.size() % d==0) ? W.size() / d : throw std::runtime_error("len(W) % d != 0");

  cpp_AnnoyBMU(this->BMU, this->QE, X, W, d, nBMU, nAnnoyTrees);

  this->flag_BMU = true;
  this->flag_Recall = false;
  this->flag_RecallLabels = false;

  return;
}

inline void VQRecall::Recall(const std::vector<double>& X, const std::vector<double>& W, const std::vector<int>& XL) {

  this->calc_BMU(X, W);


  cpp_AnnoyRecall(this->RF, this->RF_Size,
                  this->CADJi, this->CADJj, this->CADJ,
                  this->RFL_Dist, this->RFL, this->RFL_Purity, this->RFL_Purity_UOA, this->RFL_Purity_WOA,
                  this->BMU, this->QE, this->N, this->M, XL);
}

#endif

