package com.optiflow.service;

import java.time.LocalDateTime;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import com.optiflow.domain.ElectricityCostPredict;
import com.optiflow.domain.Reservoir;
import com.optiflow.dto.ElectricityCostPredictItemDto;
import com.optiflow.dto.ElectricityCostPredictRequestDto;
import com.optiflow.dto.ElectricityCostPredictResponseDto;
import com.optiflow.persistence.ElectricityCostPredictRepository;
import com.optiflow.persistence.ReservoirRepository;

import lombok.Data;

@Service
public class ElectricityCostPredictService {

	@Value("${fastapi.url}")
	private String fastapiUrl;

	@Autowired
	private ElectricityCostPredictRepository costRepo;

	@Autowired
	private ReservoirRepository reservoirRepo;

	public ElectricityCostPredict savePredict(String name, String datetime, List<ElectricityCostPredictItemDto> truth,
			List<ElectricityCostPredictItemDto> optimization) {
		ElectricityCostPredict predict = new ElectricityCostPredict();
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(name);
		int reservoirId = reservoirOptional.get().getReservoirId();
		Optional<Reservoir> reservoirEntity = reservoirRepo.findById(reservoirId);

		predict.setDatetime(datetime);
		predict.setTruth(truth);
		predict.setOptimization(optimization);
		predict.setReservoirId(reservoirEntity.get());
		return costRepo.save(predict);
	}

	public ElectricityCostPredictResponseDto getPrediction(ElectricityCostPredictRequestDto requestDto) {

		String datetime = requestDto.getDatetime();
		String name = requestDto.getName();

		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(name);
		int reservoirId = reservoirOptional.get().getReservoirId();
		Optional<Reservoir> reservoirEntity = reservoirRepo.findById(reservoirId);
		// DB에서 기존 예측 데이터 조회
		Optional<ElectricityCostPredict> existingPrediction = costRepo.findByDatetimeAndReservoirId(datetime,
				reservoirEntity.get());

		if (existingPrediction.isPresent()) {
			ElectricityCostPredictResponseDto responseDto = new ElectricityCostPredictResponseDto();
			responseDto.setTruth(existingPrediction.get().getTruth());
			responseDto.setOptimization(existingPrediction.get().getOptimization());
			return responseDto;
		} else {
			// DB에 데이터가 없으면 FastAPI 호출 및 DB 저장
			RestTemplate restTemplate = new RestTemplate(); // 필요하다면 빈 주입 방식으로 변경 고려
			HttpHeaders headers = new HttpHeaders();
			headers.setContentType(MediaType.APPLICATION_JSON);
			try {

				ElectricityCostPredictResponseDto responseDto = restTemplate.postForObject(fastapiUrl + "/api/cost",
						requestDto, // 요청 본문: requestDto 객체 직접 전달
						ElectricityCostPredictResponseDto.class);

				savePredict(name, datetime, responseDto.getTruth(), responseDto.getOptimization());
				return responseDto;

			} catch (RestClientException e) {
				System.err.println("API 호출 오류 발생: " + e.getMessage());
				e.printStackTrace();
				return new ElectricityCostPredictResponseDto();
			}
		}
	}

	// 월별 데이터 조회 메소드 추가
	public Map<String, List<?>> getDailyCostData(String name, String datetime) {
		Optional<Reservoir> reservoir = reservoirRepo.findByName(name);
		YearMonth yearMonth = YearMonth.parse(datetime, DateTimeFormatter.ofPattern("yyyy-MM"));

		LocalDateTime startDateTime = yearMonth.atDay(1).atStartOfDay(); // 월 시작 날짜
		LocalDateTime endDateTime = yearMonth.plusMonths(1).atDay(1).atStartOfDay();

		String startTime = startDateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
		String endTime = endDateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
		List<ElectricityCostPredict> dailyPredictions = costRepo
				.findByReservoirIdAndDatetimeBetweenOrderByDatetimeAsc(reservoir.get(), startTime, endTime);

		// 일별 합계 계산
		Map<LocalDateTime, DailyElectricityCostDto> dailyDataMap = new LinkedHashMap<>(); // 순서 보장 LinkedHashMap 사용

		for (ElectricityCostPredict prediction : dailyPredictions) {
			LocalDateTime date = LocalDateTime.parse(prediction.getDatetime());

			int dailyTruthSum = prediction.getTruth().stream().mapToInt(ElectricityCostPredictItemDto::getValue).sum();
			int dailyOptiSum = prediction.getOptimization().stream().mapToInt(ElectricityCostPredictItemDto::getValue)
					.sum();

			DailyElectricityCostDto dailyDto = dailyDataMap.getOrDefault(date, new DailyElectricityCostDto(date));
			dailyDto.addTruthSum(dailyTruthSum);
			dailyDto.addOptiSum(dailyOptiSum);
			dailyDataMap.put(date, dailyDto);
		}
		List<DailyElectricityCostDto> dailyDataList = new ArrayList<>(dailyDataMap.values()); // Map -> List 변환
		return convertToResponseMapForDaily(dailyDataList);
	}

	public Map<String, List<?>> getMonthlyCostData(String name, String datetime) {
		Optional<Reservoir> reservoir = reservoirRepo.findByName(name);
		int year = Integer.parseInt(datetime);

		LocalDateTime startDate = LocalDateTime.of(year, 1, 1, 0, 0); // 연도 시작 날짜
		LocalDateTime endDate = LocalDateTime.of(year, 12, 31, 23, 59, 59); // 연도 마지막 날짜

		String startTime = startDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
		String endTime = endDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
		List<ElectricityCostPredict> monthlyPredictions = costRepo
				.findByReservoirIdAndDatetimeBetweenOrderByDatetimeAsc(reservoir.get(), startTime, endTime);

		// 일별 합계 계산
		Map<LocalDateTime, MonthlyElectricityCostDto> monthlyDataMap = new LinkedHashMap<>(); // 순서 보장 LinkedHashMap 사용

		for (ElectricityCostPredict prediction : monthlyPredictions) {
			LocalDateTime date = LocalDateTime.parse(prediction.getDatetime());

			LocalDateTime monthKey = LocalDateTime.of(date.getYear(), date.getMonth(), 1, 0, 0, 0);

			int mothlyTruthSum = prediction.getTruth().stream().mapToInt(ElectricityCostPredictItemDto::getValue).sum();
			int mothlyOptiSum = prediction.getOptimization().stream().mapToInt(ElectricityCostPredictItemDto::getValue)
					.sum();

			MonthlyElectricityCostDto monthlyDto = monthlyDataMap.getOrDefault(monthKey,
					new MonthlyElectricityCostDto(monthKey));
			monthlyDto.addTruthSum(mothlyTruthSum);
			monthlyDto.addOptiSum(mothlyOptiSum);
			monthlyDataMap.put(monthKey, monthlyDto);
		}

		List<MonthlyElectricityCostDto> monthlyDataList = new ArrayList<>(monthlyDataMap.values()); // Map -> List 변환
		return convertToResponseMapForMonthly(monthlyDataList);
	}

	public Map<String, List<?>> convertToResponseMap(List<ElectricityCostPredictResponseDto> datas) {
		Map<String, List<?>> responseMap = new HashMap<>();

		List<Integer> truthValueList = new ArrayList<>();
		List<Integer> optimizationValueList = new ArrayList<>();
		List<String> timeList = new ArrayList<>();
		List<Double> percentList = new ArrayList<>();

		if (!datas.isEmpty()) {
			ElectricityCostPredictResponseDto data = datas.get(0);

			for (ElectricityCostPredictItemDto item : data.getTruth()) {
				truthValueList.add(item.getValue());
				timeList.add(item.getTime().toString());
			}
			for (ElectricityCostPredictItemDto item : data.getOptimization()) {
				optimizationValueList.add(item.getValue());
			}
			int totalTruth = truthValueList.stream().mapToInt(Integer::intValue).sum();
			int totalOpti = optimizationValueList.stream().mapToInt(Integer::intValue).sum();
			double overallPercent = totalTruth != 0 ? ((totalTruth - totalOpti) / (double) totalTruth) * 100 : 0;
			percentList.add(overallPercent);
		}

		responseMap.put("truth", truthValueList);
		responseMap.put("optimization", optimizationValueList);
		responseMap.put("date", timeList);
		responseMap.put("percent", percentList);

		return responseMap;
	}

	public Map<String, List<?>> convertToResponseMapForDaily(List<DailyElectricityCostDto> dailyDatas) {
		Map<String, List<?>> responseMap = new HashMap<>();

		List<Integer> truthValueList = new ArrayList<>();
		List<Integer> optimizationValueList = new ArrayList<>();
		List<String> dateList = new ArrayList<>(); // timeList -> dateList 로 변경
		List<Double> percentList = new ArrayList<>();

		if (!dailyDatas.isEmpty()) {
			DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
			for (DailyElectricityCostDto dailyData : dailyDatas) { // dailyDatas 리스트 순회
				truthValueList.add(dailyData.getTruthSum()); // 일별 합계 값 추가
				optimizationValueList.add(dailyData.getOptiSum()); // 일별 합계 값 추가
				dateList.add(dailyData.getDate().format(formatter)); // 날짜 추가
			}

			int totalTruth = truthValueList.stream().mapToInt(Integer::intValue).sum();
			int totalOpti = optimizationValueList.stream().mapToInt(Integer::intValue).sum();
			double overallPercent = totalTruth != 0 ? ((totalTruth - totalOpti) / (double) totalTruth) * 100 : 0;
			percentList.add(overallPercent);
		}

		responseMap.put("truth", truthValueList);
		responseMap.put("optimization", optimizationValueList);
		responseMap.put("date", dateList); // time -> date 로 key 변경
		responseMap.put("percent", percentList);

		return responseMap;
	}

	public Map<String, List<?>> convertToResponseMapForMonthly(List<MonthlyElectricityCostDto> monthlyDatas) {
		Map<String, List<?>> responseMap = new HashMap<>();

		List<Integer> truthValueList = new ArrayList<>();
		List<Integer> optimizationValueList = new ArrayList<>();
		List<String> monthList = new ArrayList<>(); // 월 정보 리스트 추가
		List<Double> percentList = new ArrayList<>();

		if (!monthlyDatas.isEmpty()) {
			DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM");

			for (MonthlyElectricityCostDto monthlyData : monthlyDatas) {
				truthValueList.add(monthlyData.getTruthSum());
				optimizationValueList.add(monthlyData.getOptiSum());
				monthList.add(monthlyData.getDate().format(formatter));
			}

			int totalTruth = truthValueList.stream().mapToInt(Integer::intValue).sum();
			int totalOpti = optimizationValueList.stream().mapToInt(Integer::intValue).sum();
			double overallPercent = totalTruth != 0 ? ((totalTruth - totalOpti) / (double) totalTruth) * 100 : 0;
			percentList.add(overallPercent);
		}

		responseMap.put("truth", truthValueList);
		responseMap.put("optimization", optimizationValueList);
		responseMap.put("date", monthList); // time -> date 로 key 변경
		responseMap.put("percent", percentList);

		return responseMap;
	}

	@Data
	private static class DailyElectricityCostDto {
		private LocalDateTime date;
		private int truthSum = 0;
		private int optiSum = 0;

		public DailyElectricityCostDto(LocalDateTime date2) {
			this.date = date2;
		}

		public void addTruthSum(int sum) {
			this.truthSum += sum;
		}

		public void addOptiSum(int sum) {
			this.optiSum += sum;
		}
	}

	@Data
	private static class MonthlyElectricityCostDto {
		private LocalDateTime date;
		private int truthSum = 0;
		private int optiSum = 0;

		public MonthlyElectricityCostDto(LocalDateTime date3) {
			this.date = date3;
		}

		public void addTruthSum(int sum) {
			this.truthSum += sum;
		}

		public void addOptiSum(int sum) {
			this.optiSum += sum;
		}
	}
}
