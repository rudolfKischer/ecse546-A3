from enum import IntEnum

import taichi as ti
import taichi.math as tm

from .scene_data import SceneData
from .camera import Camera
from .ray import Ray, HitData
from .sampler import UniformSampler, BRDF, MicrofacetBRDF
from .materials import Material


@ti.data_oriented
class A1Renderer:

    # Enumerate the different shading modes
    class ShadeMode(IntEnum):
        HIT = 1
        TRIANGLE_ID = 2
        DISTANCE = 3
        BARYCENTRIC = 4
        NORMAL = 5
        MATERIAL_ID = 6

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.scene_data = scene_data

        self.shade_mode = ti.field(shape=(), dtype=int)
        self.set_shade_hit()

        # Distance at which the distance shader saturates
        self.max_distance = 10.

        # Numbers used to generate colors for integer index values
        self.r = 3.14159265
        self.b = 2.71828182
        self.g = 6.62607015


    def set_shade_hit(self):          self.shade_mode[None] = self.ShadeMode.HIT
    def set_shade_triangle_ID(self):  self.shade_mode[None] = self.ShadeMode.TRIANGLE_ID
    def set_shade_distance(self):     self.shade_mode[None] = self.ShadeMode.DISTANCE
    def set_shade_barycentrics(self): self.shade_mode[None] = self.ShadeMode.BARYCENTRIC
    def set_shade_normal(self):       self.shade_mode[None] = self.ShadeMode.NORMAL
    def set_shade_material_ID(self):  self.shade_mode[None] = self.ShadeMode.MATERIAL_ID


    @ti.kernel
    def render(self):
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y)
            color = self.shade_ray(primary_ray)
            self.canvas[x,y] = color


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        hit_data = self.scene_data.ray_intersector.query_ray(ray)
        color = tm.vec3(0)
        if   self.shade_mode[None] == int(self.ShadeMode.HIT):         color = self.shade_hit(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.TRIANGLE_ID): color = self.shade_triangle_id(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.DISTANCE):    color = self.shade_distance(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.BARYCENTRIC): color = self.shade_barycentric(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.NORMAL):      color = self.shade_normal(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.MATERIAL_ID): color = self.shade_material_id(hit_data)
        return color
       

    @ti.func
    def shade_hit(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            if not hit_data.is_backfacing:
                color = tm.vec3(1)
            else: 
                color = tm.vec3([0.5,0,0])
        return color


    @ti.func
    def shade_triangle_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            triangle_id = hit_data.triangle_id + 1 # Add 1 so that ID 0 is not black
            r = triangle_id*self.r % 1
            g = triangle_id*self.g % 1
            b = triangle_id*self.b % 1
            color = tm.vec3(r,g,b)
        return color


    @ti.func
    def shade_distance(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            d = tm.clamp(hit_data.distance / self.max_distance, 0,1)
            color = tm.vec3(d)
        return color


    @ti.func
    def shade_barycentric(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            u = hit_data.barycentric_coords[0]
            v = hit_data.barycentric_coords[1]
            w = 1. - u - v
            color = tm.vec3(u,v,w)
        return color


    @ti.func
    def shade_normal(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            normal = hit_data.normal
            color = (normal + 1.) / 2.  # Scale to range [0,1]
        return color


    @ti.func
    def shade_material_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            material_id = hit_data.material_id + 1 # Add 1 so that ID 0 is not black
            r = material_id*self.r % 1
            g = material_id*self.g % 1
            b = material_id*self.b % 1
            color = tm.vec3(r,g,b)
        return color

@ti.data_oriented
class A2Renderer:

    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        MICROFACET = 3

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.RAY_OFFSET = 1e-4

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()


    def set_sample_uniform(self):    self.sample_mode[None] = self.SampleMode.UNIFORM
    def set_sample_brdf(self):       self.sample_mode[None] = self.SampleMode.BRDF
    def set_sample_microfacet(self): self.sample_mode[None] = self.SampleMode.MICROFACET


    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y, jitter=True)
            color = self.shade_ray(primary_ray)
            
            self.canvas[x,y] += (color - self.canvas[x,y]) / self.iter_counter[None]

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)


    @ti.func
    def shade_triangle_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        mat = self.scene_data.material_library.materials[hit_data.material_id]
        diffuse = mat.Kd
        if hit_data.is_hit:
            triangle_id = hit_data.triangle_id + 1 # Add 1 so that ID 0 is not black
            r = triangle_id*diffuse[0] % 1
            g = triangle_id*diffuse[1] % 1
            b = triangle_id*diffuse[2] % 1
            color = tm.vec3(r,g,b)
        return color
    
    @ti.func
    def phong_specular_brdf(self, x: HitData, w_i: tm.vec3, w_o: tm.vec3) -> tm.vec3:
        w_o = w_o.normalized()
        w_i = w_i.normalized()
        mat = self.scene_data.material_library.materials[x.material_id]
        rho_s = mat.Kd
        alpha = mat.Ns
        w_r = (2 * w_o.dot(x.normal) * x.normal - w_o).normalized()
        specular = (rho_s * (alpha + 1) / (2 * tm.pi)) * max(tm.pow(w_r.dot(w_i), alpha),0)
        return specular
    
    @ti.func
    def phong_diffuse_brdf(self, x: HitData, w_i: tm.vec3, w_o: tm.vec3) -> tm.vec3:
        mat = self.scene_data.material_library.materials[x.material_id]
        rho_d = mat.Kd
        diffuse = rho_d / tm.pi
        return diffuse

    @ti.func
    def phong_brdf(self, x: HitData, w_i: tm.vec3, w_o: tm.vec3) -> tm.vec3:
        mat = self.scene_data.material_library.materials[x.material_id]
        alpha = mat.Ns
        result = tm.vec3(0)
        if alpha <= 1:
            result = self.phong_diffuse_brdf(x, w_i, w_o)
        else:
            result = self.phong_specular_brdf(x, w_i, w_o)
        return result
        

    @ti.func
    def visibility(self, x: HitData, r_o: Ray) -> tm.vec3:
        hit_data = self.scene_data.ray_intersector.query_ray(r_o)
        result = tm.vec3(1)
        if hit_data.is_hit:
            result = tm.vec3(0)
        return result
    
    @ti.func
    def reflect(self, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        w_i = w_i.normalized()
        normal = normal.normalized()
        return (2 * w_i.dot(normal) * normal - w_i).normalized()
     

    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)


        # if the ray does no hit anything, we return the color of the environment map
        hit_data = self.scene_data.ray_intersector.query_ray(ray)
        mat = self.scene_data.material_library.materials[hit_data.material_id]
        env_map = self.scene_data.environment

        if not hit_data.is_hit:
            color = env_map.query_ray(ray)
        else:
          if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
              # ray contains a t, which is the distance to the closest object
              # and the object that was hit, which tells us the material of the object
              # Lo(X, w_o)_i = Le(X, w_o)_i * V(X, w_o)_i * f_r(X, w_i, w_o)_i max(0, dot(w_i, N)) / p(w_i)

              w_i = -ray.direction
              u_sampler = UniformSampler()
              w_o = u_sampler.sample_direction()
              # w_o is in the on the sphere, but we want it to be the normal hemisphere
              # we can do this by taking the dot product of the normal and the direction
              # if the dot product is negative, we flip the direction across the normal
              normal = hit_data.normal
              if w_o.dot(normal) < 0:
                  w_o = -w_o

              cos_theta = max(w_o.dot(normal), 0)
              p_u = u_sampler.evaluate_probability() * 2 # multiply by 2 because we are sampling hemisphere, not sphere

              #incoming light
              p = hit_data.distance * ray.direction + ray.origin
              r_i = Ray(p + 10 * self.RAY_OFFSET * w_o, w_o.normalized())
              l_i = env_map.query_ray(r_i)
              V_x = self.visibility(hit_data, r_i)
              color = l_i * V_x  * self.phong_brdf(hit_data, w_i, w_o) * cos_theta / p_u

                  
              
          elif self.sample_mode[None] == int(self.SampleMode.BRDF):
              
              w_i = -ray.direction.normalized()
              normal = hit_data.normal.normalized()
              brdf = BRDF()
              w_o = brdf.sample_direction(mat, w_i, normal).normalized()

              p = hit_data.distance * ray.direction.normalized() + ray.origin
              r_i = Ray(p + self.RAY_OFFSET * w_o, w_o.normalized())
              l_i = env_map.query_ray(r_i)
              V_x = self.visibility(hit_data, r_i)
              color = l_i * brdf.evaluate_brdf(mat, w_o, w_i, hit_data.normal.normalized()) * V_x
              # cos_theta = max(w_o.dot(normal), 0)
              # TODO: USING THis value causes overflow and underflow errors with large alpha
              # p_u = brdf.evaluate_probability(mat, w_o, w_i, hit_data.normal.normalized())
              # color =  (l_i * V_x * self.phong_brdf(hit_data, w_i, w_o)) * cos_theta  / p_u 
                            
          # Implement Microfacet BRDF Sampling
          elif self.sample_mode[None] == int(self.SampleMode.MICROFACET):
              
              w_o = -ray.direction.normalized()
              normal = hit_data.normal.normalized()
              microfacetbrdf = MicrofacetBRDF()
              w_m = microfacetbrdf.sample_direction(mat, w_o, normal).normalized()
              w_i = self.reflect(w_o, w_m).normalized()
              

              
              p = hit_data.distance * ray.direction.normalized() + ray.origin
              r_i = Ray(p + 100 * self.RAY_OFFSET * w_i, w_i.normalized())
              l_i = env_map.query_ray(r_i)
              V_x = self.visibility(hit_data, r_i)


              
              f_r_microfacet = microfacetbrdf.evaluate_brdf(mat, w_o, w_i, w_m, normal)
              p_microfacet = microfacetbrdf.evaluate_probability(mat, w_o, w_i, w_m, normal)
              cos_theta = max(w_i.dot(normal), 0)

              color = f_r_microfacet / p_microfacet * l_i * V_x * cos_theta
        
        
        return color


@ti.data_oriented
class EnvISRenderer:
    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        ENVMAP = 2
    
    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.width = width
        self.height = height
        
        self.camera = Camera(width=width, height=height)
        self.count_map = ti.field(dtype=float, shape=(width, height))
        
        self.background = ti.Vector.field(n=3, dtype=float, shape=(width, height))

        self.scene_data = scene_data
        self.sample_mode = ti.field(shape=(), dtype=int)

        self.set_sample_uniform()


    def set_sample_uniform(self): 
        self.sample_mode[None] = self.SampleMode.UNIFORM
    def set_sample_envmap(self):    
        self.sample_mode[None] = self.SampleMode.ENVMAP

    @ti.func
    def render_background(self, x: int, y: int) -> tm.vec3:
        uv_x, uv_y = float(x)/self.width, float(y)/self.height
        uv_x, uv_y = uv_x*self.scene_data.environment.x_resolution, uv_y*self.scene_data.environment.y_resolution
        
        background = self.scene_data.environment.image[int(uv_x), int(uv_y)]
            

        return background


    @ti.kernel
    def render_background(self):
        for x,y in ti.ndrange(self.width, self.height):
            uv_x, uv_y = float(x)/float(self.width), float(y)/float(self.height)
            uv_x, uv_y = uv_x*self.scene_data.environment.x_resolution, uv_y*self.scene_data.environment.y_resolution
            color = self.scene_data.environment.image[int(uv_x), int(uv_y)]

            self.background[x,y] = color

    @ti.kernel
    def sample_env(self, samples: int):
        for _ in ti.ndrange(samples):
            if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
                x = int(ti.random() * self.width)
                y = int(ti.random() * self.height)


                self.count_map[x,y] += 1.0
                
            elif self.sample_mode[None] == int(self.SampleMode.ENVMAP):
                sampled_phi_theta = self.scene_data.environment.importance_sample_envmap()
                x = sampled_phi_theta[0] * self.width
                y = sampled_phi_theta[1] * self.height

                self.count_map[int(x), int(y)] += 1.0
    
    @ti.kernel
    def reset(self):
        self.count_map.fill(0.)


@ti.data_oriented
class A3Renderer:

    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        LIGHT = 3
        MIS = 4

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.RAY_OFFSET = 1e-6

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.canvas_postprocessed = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data
        self.a2_renderer = A2Renderer(width=self.width, height=self.height, scene_data=self.scene_data)
        
        self.mis_plight = ti.field(dtype=float, shape=())
        self.mis_pbrdf = ti.field(dtype=float, shape=())

        self.mis_plight[None] = 0.5
        self.mis_pbrdf[None] = 0.5

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()


    def set_sample_uniform(self): 
        self.sample_mode[None] = self.SampleMode.UNIFORM
        self.a2_renderer.set_sample_uniform()
    def set_sample_brdf(self):    
        self.sample_mode[None] = self.SampleMode.BRDF
        self.a2_renderer.set_sample_brdf()
    def set_sample_light(self):    self.sample_mode[None] = self.SampleMode.LIGHT
    def set_sample_mis(self):    self.sample_mode[None] = self.SampleMode.MIS


    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1.0
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x,y] += (color - self.canvas[x,y])/self.iter_counter[None]
    
    @ti.kernel
    def postprocess(self):
        for x,y in ti.ndrange(self.width, self.height):
            self.canvas_postprocessed[x, y] = tm.pow(self.canvas[x, y], tm.vec3(1.0 / 2.2))
            self.canvas_postprocessed[x, y] = tm.clamp(self.canvas_postprocessed[x, y], xmin=0.0, xmax=1.0)

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)
        if self.sample_mode[None] == int(self.SampleMode.UNIFORM) or self.sample_mode[None] == int(self.SampleMode.BRDF):
            # Uniform or BRDF just calls the A2 renderer
            # TODO: Implement Mesh Light support for your A2 renderer
            color = self.a2_renderer.shade_ray(ray)
        else:
            if self.sample_mode[None] == int(self.SampleMode.LIGHT):
                # TODO: Implement Light Importance Sampling
                pass        
            if self.sample_mode[None] == int(self.SampleMode.MIS):
                # TODO: Implement MIS
                pass     
                     
        return color

