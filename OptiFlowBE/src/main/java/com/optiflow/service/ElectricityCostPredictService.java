package com.optiflow.service;

import java.util.ArrayList;
import java.util.HashMap;
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

@Service
public class ElectricityCostPredictService {

	@Value("${fastapi.url}")
	private String fastapiUrl;

	@Autowired
	private ElectricityCostPredictRepository costRepo;

	@Autowired
	private ReservoirRepository reservoirRepo;

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

	public ElectricityCostPredict savePredict(String name, String datetime,
			List<ElectricityCostPredictItemDto> truth, List<ElectricityCostPredictItemDto> optimization) {
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
		responseMap.put("time", timeList);
		responseMap.put("percent", percentList);

		return responseMap;
	}
}
