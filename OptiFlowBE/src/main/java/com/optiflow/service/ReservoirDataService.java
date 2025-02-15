package com.optiflow.service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.Year;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.optiflow.domain.Reservoir;
import com.optiflow.domain.ReservoirData;
import com.optiflow.persistence.ReservoirDataRepository;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class ReservoirDataService {

	@Autowired
	private ReservoirDataRepository reservoirDataRepo;

	public List<ReservoirData> findByObservationTimeRange(Reservoir reservoir, LocalDateTime startTime,
			LocalDateTime endTime) {
		return reservoirDataRepo.findByReservoirIdAndObservationTimeBetween(reservoir, startTime, endTime);
	}

	public List<ReservoirData> findByObservationTime(LocalDateTime observationTime) {
		return reservoirDataRepo.findByObservationTime(observationTime);
	}

	public Map<String, List<?>> findDailyStatsByReservoirId(int reservoirId) {
		List<Object[]> results = reservoirDataRepo.findDailyStatsByReservoirId(reservoirId);
		return convertHourlyResultsToResponseMap(results);
	}

	// 시간별 통계 (일별 기준)
	public Map<String, List<?>> findHourlyStatsByDailyObservationTimeAndReservoirId(String date, int reservoirId) {
		LocalDate startDate = LocalDate.parse(date, DateTimeFormatter.ofPattern("yyyy-MM-dd"));
		LocalDate endDate = startDate; // 당일 하루

		List<Object[]> results = reservoirDataRepo.findHourlyStatsByDailyObservationTimeAndReservoirId(
				startDate.toString() + " 00:00:00", endDate.toString() + " 23:59:59", reservoirId);
		return convertHourlyResultsToResponseMap(results);
	}

	// 일별 통계 (월별 기준)
	public Map<String, List<?>> findDailyStatsByMonthlyObservationTimeAndReservoirId(String month, int reservoirId) {
		YearMonth yearMonth = YearMonth.parse(month, DateTimeFormatter.ofPattern("yyyy-MM"));
		LocalDate startDate = yearMonth.atDay(1); // 월 시작일
		LocalDate endDate = yearMonth.atEndOfMonth(); // 월 마지막일

		LocalDateTime startDateTime = startDate.atTime(LocalTime.MIDNIGHT);
		LocalDateTime endDateTime = endDate.atTime(LocalTime.of(23, 59, 59));

		String startTime = startDateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
		String endTime = endDateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

		List<Object[]> results = reservoirDataRepo.findDailyStatsByMonthlyObservationTimeAndReservoirId(startTime,
				endTime, reservoirId);
		return convertDailyResultsToResponseMap(results);
	}

	// 월별 통계 (년별 기준)
	public Map<String, List<?>> findMonthlyStatsByYearlyObservationTimeAndReservoirId(String year, int reservoirId) {
		Year yearObj = Year.parse(year, DateTimeFormatter.ofPattern("yyyy"));
		LocalDate startMonth = yearObj.atMonth(1).atDay(1);
		LocalDate endMonth = yearObj.atMonth(12).atEndOfMonth();
		List<Object[]> results = reservoirDataRepo.findMonthlyStatsByYearlyObservationTimeAndReservoirId(
				startMonth.toString(), endMonth.toString(), reservoirId);
		return convertMonthlyResultsToResponseMap(results);
	}

    public Map<String, List<?>> convertHourlyResultsToResponseMap(List<Object[]> results) {
        Map<String, List<?>> responseMap = new HashMap<>();
        List<Double> outputList = results.stream()
                .map(result -> ((Number) result[3]).doubleValue())
                .collect(Collectors.toList());
        responseMap.put("output", outputList);
        return responseMap;
    }
    
    public Map<String, List<?>> convertDailyResultsToResponseMap(List<Object[]> results) {
        Map<String, List<?>> responseMap = new HashMap<>();
        List<Double> outputList = results.stream()
                .map(result -> ((Number) result[3]).doubleValue() /24)
                .collect(Collectors.toList());
        responseMap.put("output", outputList);
        return responseMap;
    }
    
    public Map<String, List<?>> convertMonthlyResultsToResponseMap(List<Object[]> results) {
        Map<String, List<?>> responseMap = new HashMap<>();
        List<Double> outputList = results.stream()
                .map(result -> ((Number) result[3]).doubleValue() /12)
                .collect(Collectors.toList());
        responseMap.put("output", outputList);
        return responseMap;
    }
}

