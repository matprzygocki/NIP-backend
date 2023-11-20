package com.example.airestappsidecar;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class DefaultController {

    @RequestMapping("/")
    @ResponseBody
    public String defaultMapping() {
        return "Witaj! To jest domyślna strona aplikacji.";
    }
}
