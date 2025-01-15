package com.optiflow.handler;

import java.io.IOException;

import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.stereotype.Component;

import com.optiflow.util.JWTUtil;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@RequiredArgsConstructor
@Slf4j
@Component
public class SuccessHandler implements AuthenticationSuccessHandler{
	
	public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response,
			Authentication authentication) throws IOException, ServletException {
//        String username = authentication.getName();
//
//        String jwtToken = JWTUtil.getJWT(username);
//        String redirectUrl = String.format("http://localhost:3000/member", jwtToken);
        
        String redirectUrl = String.format("http://localhost:3000/member");
        response.sendRedirect(redirectUrl);
	}

}
