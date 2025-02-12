package com.optiflow.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.optiflow.domain.Inquiry;
import com.optiflow.persistence.InquiryRepository;

@Service
public class InquiryService {
	
	@Autowired
	private InquiryRepository inquiryRepo;
	
	public List<Inquiry> getAllInquirys(){
		return inquiryRepo.findAllByOrderByInquiryIdDesc();
	}
    public Inquiry saveInquiry(Inquiry inquiry) {
        return inquiryRepo.save(inquiry);
    }
    
    public List<Inquiry> getUnapprovedInquiries() {
        return inquiryRepo.findAllByApprovedIsFalseOrderByInquiryIdDesc();
    }
    
    public List<Inquiry> getUnconfirmedInquiries() {
        return inquiryRepo.findAllByStaffConfirmedIsFalseOrderByInquiryIdDesc();
    }
    
    public Inquiry getInquiryById(Long inquiryId) {
        Optional<Inquiry> inquiryOptional = inquiryRepo.findById(inquiryId);
        return inquiryOptional.orElse(null); // 문의가 없으면 null 반환
    }
}
