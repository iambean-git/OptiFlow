package com.optiflow.service;

import java.util.List;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.optiflow.domain.ReservoirData;
import com.optiflow.dto.ReservoirStats;
import com.optiflow.persistence.ReservoirDataReposotory;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class ReservoirDataService {

	@Autowired
	private ReservoirDataReposotory reservoirDataRepo;

	public List<ReservoirData> getAllReservoirData() {
		return reservoirDataRepo.findAll();
	}

//	public List<ReservoirStats> getDailyStatsByReservoirId(int reservoirId) {
//		List<ReservoirStats> stats = reservoirDataRepo.findDailyStatsByReservoirId(reservoirId);
//		log.info("Daily Stats by Reservoir ID from Repository (ID: {}): {}", reservoirId, stats); // 로그 출력
//		return stats;
//	}


//	public List<ReservoirStats> getMonthlyStatsByReservoirId(Long reservoirId) {
//		List<ReservoirStats> stats = reservoirDataRepo.findMonthlyStatsByReservoirId(reservoirId);
//		log.info("Monthly Stats by Reservoir ID from Repository (ID: {}): {}", reservoirId, stats); // 로그 출력
//		return stats;
//	}
//
//	public List<ReservoirStats> getYearlyStatsByReservoirId(Long reservoirId) {
//		List<ReservoirStats> stats = reservoirDataRepo.findYearlyStatsByReservoirId(reservoirId);
//		log.info("Yearly Stats by Reservoir ID from Repository (ID: {}): {}", reservoirId, stats); // 로그 출력
//		return stats;
//	}
//	public List<Object> getReservoirStatsByGroup(String groupBy) {
//        String pattern;
//        switch (groupBy.toLowerCase()) {
//            case "daily":
//                pattern = "%Y-%m-%d"; // 일별
//                break;
//            case "monthly":
//                pattern = "%Y-%m";   // 월별
//                break;
//            case "yearly":
//                pattern = "%Y";      // 연별
//                break;
//            default:
//                throw new IllegalArgumentException("Invalid groupBy value: " + groupBy);
//        }
//        return reservoirDataRepo.getGroupedReservoirStats(pattern);
//    }
//	
	public List<ReservoirStats> findDailyStatsByReservoirId(int reservoirId) {
		List<Object[]> results = reservoirDataRepo.findDailyStatsByReservoirId(reservoirId);
		List<ReservoirStats> stats = results.stream()
		    .map(result -> new ReservoirStats(
		        (String) result[0],          // observationTime
		        (int) result[1],            // reservoirId
		        ((Double)result[2]), // totalInput
		        ((Double)result[3]), // totalOutput
		        ((Double)result[4])  // avgHeight
		    ))
		    .collect(Collectors.toList());
		return stats;
	}
	
	public List<ReservoirStats> findMonthlyStatsByReservoirId(int reservoirId) {
		List<Object[]> results = reservoirDataRepo.findMonthlyStatsByReservoirId(reservoirId);
		List<ReservoirStats> stats = results.stream()
		    .map(result -> new ReservoirStats(
		        (String) result[0],          // observationTime
		        (int) result[1],            // reservoirId
		        ((Double)result[2]), // totalInput
		        ((Double)result[3]), // totalOutput
		        ((Double)result[4])  // avgHeight
		    ))
		    .collect(Collectors.toList());
		return stats;
	}
}
