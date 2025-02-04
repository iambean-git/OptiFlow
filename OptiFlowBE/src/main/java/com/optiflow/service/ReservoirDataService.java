package com.optiflow.service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.Year;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.optiflow.domain.ReservoirData;
import com.optiflow.dto.ReservoirStats;
import com.optiflow.persistence.ReservoirDataRepository;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class ReservoirDataService {

	@Autowired
	private ReservoirDataRepository reservoirDataRepo;
	
	public List<ReservoirData> findByObservationTime(LocalDateTime observationTime){
		return reservoirDataRepo.findByObservationTime(observationTime);
	}
 
	public List<ReservoirStats> findDailyStatsByReservoirId(int reservoirId) {
		List<Object[]> results = reservoirDataRepo.findDailyStatsByReservoirId(reservoirId);
		return convertObjectArraysToReservoirStats(results);
	}
	
	 // 시간별 통계 (일별 기준)
    public List<ReservoirStats> findHourlyStatsByDailyObservationTimeAndReservoirId(String date, int reservoirId) {
        LocalDate startDate = LocalDate.parse(date, DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        LocalDate endDate = startDate; // 당일 하루

        List<Object[]> results = reservoirDataRepo.findHourlyStatsByDailyObservationTimeAndReservoirId(
                startDate.toString() + " 00:00:00",
                endDate.toString() + " 23:59:59",
                reservoirId);
        return convertObjectArraysToReservoirStats(results);
    }

    // 일별 통계 (월별 기준)
    public List<ReservoirStats> findDailyStatsByMonthlyObservationTimeAndReservoirId(String month, int reservoirId) {
        YearMonth yearMonth = YearMonth.parse(month, DateTimeFormatter.ofPattern("yyyy-MM"));
        LocalDate startDate = yearMonth.atDay(1); // 월 시작일
        LocalDate endDate = yearMonth.atEndOfMonth(); // 월 마지막일
        
        LocalDateTime startDateTime = startDate.atTime(LocalTime.MIDNIGHT);
        LocalDateTime endDateTime = endDate.atTime(LocalTime.of(23, 59, 59));

        String startTime = startDateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        String endTime = endDateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

        List<Object[]> results = reservoirDataRepo.findDailyStatsByMonthlyObservationTimeAndReservoirId(
                startTime,
                endTime,
                reservoirId);
        return convertObjectArraysToReservoirStats(results);
    }

    // 월별 통계 (년별 기준)
    public List<ReservoirStats> findMonthlyStatsByYearlyObservationTimeAndReservoirId(String year, int reservoirId) {
        Year yearObj = Year.parse(year, DateTimeFormatter.ofPattern("yyyy"));
        LocalDate startMonth = yearObj.atMonth(1).atDay(1);
        LocalDate endMonth = yearObj.atMonth(12).atEndOfMonth();
        List<Object[]> results = reservoirDataRepo.findMonthlyStatsByYearlyObservationTimeAndReservoirId(
                startMonth.toString(),
                endMonth.toString(),
                reservoirId);
        return convertObjectArraysToReservoirStats(results);
    }


    private List<ReservoirStats> convertObjectArraysToReservoirStats(List<Object[]> results) {
        return results.stream()
                .map(result -> new ReservoirStats(
                        (String) result[0], 				// observationTime
                        (int) result[1],            		// reservoirId
                        ((Number) result[2]).doubleValue(), // totalInput
                        ((Number) result[3]).doubleValue(), // totalOutput
                        ((Number) result[4]).doubleValue()  // avgHeight
                ))
                .collect(Collectors.toList());
    }
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
