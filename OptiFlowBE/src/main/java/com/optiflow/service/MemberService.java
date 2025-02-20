package com.optiflow.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.optiflow.domain.Member;
import com.optiflow.dto.PasswordChangeRequestDto;
import com.optiflow.persistence.MemberRepository;

import jakarta.transaction.Transactional;

@Service
public class MemberService {
	
	@Autowired
	private MemberRepository memberRepo;
	
	@Autowired
	private PasswordEncoder encoder;
	

	public Member saveMember(Member member) {
    	Member memberRegi = Member.builder()
                .username(member.getUsername())
                .password(encoder.encode(member.getPassword()))
                .role(Member.Role.Role_User)
                .build();
		
		return memberRepo.save(memberRegi);
	}

	public List<Member> getAllMembers() {
		return memberRepo.findAll();
	}

	public Optional<Member> getMemberByUsername(String username) {
		return memberRepo.findByUsername(username);
	}

	public void deleteMember(String username) {
		memberRepo.deleteByUsername(username);
	}
	
	@Transactional
    public void changePassword(PasswordChangeRequestDto requestDto) {
		String username = requestDto.getUsername();
        Optional<Member> memberOpt = memberRepo.findByUsername(username);
        
        if (!memberOpt.isPresent()) {
            throw new IllegalArgumentException("해당 사용자를 찾을 수 없습니다.");
        }
        
        Member member = memberOpt.get();

        if (!encoder.matches(requestDto.getPassword(), member.getPassword())) {
            throw new IllegalArgumentException("현재 비밀번호가 일치하지 않습니다.");
        }

        String encodedNewPassword = encoder.encode(requestDto.getNewpw());
        member.setPassword(encodedNewPassword);
        memberRepo.save(member);
    }

}
