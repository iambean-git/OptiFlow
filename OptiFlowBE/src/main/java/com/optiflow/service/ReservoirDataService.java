package com.optiflow.service;

import java.time.LocalDate;
import java.time.LocalDateTime;
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
	
	 // 시간별 통계 (일별 기준)
    public List<ReservoirStats> findHourlyStatsByDailyObservationTimeAndReservoirId(String date, int reservoirId) {
        LocalDate startDate = LocalDate.parse(date, DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        LocalDate endDate = startDate; // 당일 하루

        List<Object[]> results = reservoirDataRepo.findHourlyStatsByDailyObservationTimeAndReservoirId(
                startDate.toString() + " 00:00:00", // startTime (yyyy-MM-dd 00:00:00)
                endDate.toString() + " 23:59:59",   // endTime (yyyy-MM-dd 23:59:59)
                reservoirId);
        return convertObjectArraysToReservoirStats(results);
    }

    // 일별 통계 (월별 기준)
    public List<ReservoirStats> findDailyStatsByMonthlyObservationTimeAndReservoirId(String month, int reservoirId) {
        YearMonth yearMonth = YearMonth.parse(month, DateTimeFormatter.ofPattern("yyyy-MM"));
        LocalDate startDate = yearMonth.atDay(1); // 월 시작일
        LocalDate endDate = yearMonth.atEndOfMonth(); // 월 마지막일

        List<Object[]> results = reservoirDataRepo.findDailyStatsByMonthlyObservationTimeAndReservoirId(
                startDate.toString(), // startTime (yyyy-MM-dd)
                endDate.toString(),   // endTime (yyyy-MM-dd)
                reservoirId);
        return convertObjectArraysToReservoirStats(results);
    }

    // 월별 통계 (년별 기준)
    public List<ReservoirStats> findMonthlyStatsByYearlyObservationTimeAndReservoirId(String year, int reservoirId) {
        Year yearObj = Year.parse(year, DateTimeFormatter.ofPattern("yyyy"));
        LocalDate startMonth = yearObj.atMonth(1).atDay(1); // 1월
        LocalDate endMonth = yearObj.atMonth(12).atEndOfMonth();  // 12월
        List<Object[]> results = reservoirDataRepo.findMonthlyStatsByYearlyObservationTimeAndReservoirId(
                startMonth.toString(), // startTime (yyyy-MM)
                endMonth.toString(),   // endTime (yyyy-MM)
                reservoirId);
        return convertObjectArraysToReservoirStats(results);
    }


    private List<ReservoirStats> convertObjectArraysToReservoirStats(List<Object[]> results) {
        return results.stream()
                .map(result -> new ReservoirStats(
                        (String) result[0],          // observationTime (String으로 유지)
                        (int) result[1],            // reservoirId
                        ((Number) result[2]).doubleValue(), // totalInput (Double 또는 BigDecimal 등 Number 타입 처리)
                        ((Number) result[3]).doubleValue(), // totalOutput (Double 또는 BigDecimal 등 Number 타입 처리)
                        ((Number) result[4]).doubleValue()  // avgHeight (Double 또는 BigDecimal 등 Number 타입 처리)
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
